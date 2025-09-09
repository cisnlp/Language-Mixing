import os
import re
import json
import logging
import unicodedata
from typing import Dict, Tuple, Optional
from tqdm import tqdm

from utils_evaluation import get_latest_files


logger = logging.getLogger(__name__)

# Define multilingual role keywords
ROLE_KEYWORDS = {
    "en": {
        "knight": ["knight"],
        "knave": ["knave", "liar"]
    },
    "zh": {
        "knight": ["骑士", "英雄", "勇者"],
        "knave": ["无赖", "骗子", "恶棍"]
    },
    "ja": {
        "knight": ["騎士"],
        "knave": ["ならず者"]
    },
    "ar": {
        "knight": ["فارس", "فرسان"],
        "knave": ["محتال", "محتالة"]
    },
    "fr": {
        "knight": ["chevalier", "chevalière"],
        "knave": ["vaurien", "vaurienne"]
    },
    "hi": {
        "knight": ["शूरवीर"],
        "knave": ["धूर्त"]
    }
}


def load_name_translations(lang: str) -> Dict[str, str]:
    """Load a language's name translation dict and return reverse mapping (local name → English name)."""
    file_map = {
        "zh": "data/kk/translation_map/name_translations_zh.json",
        "ja": "data/kk/translation_map/name_translations_ja.json",
        "ar": "data/kk/translation_map/name_translations_ar.json",
        "hi": "data/kk/translation_map/name_translations_hi.json",
        "fr": "data/kk/translation_map/name_translations_fr.json",
    }

    if lang not in file_map:
        return {}

    with open(file_map[lang], "r", encoding="utf-8") as f:
        en_to_local = json.load(f)
        # Reverse mapping: local → English
        return list(en_to_local.values())
    

def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    """Extracts the final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
    # Split response to isolate assistant output
    # if "Assistant:" in solution_str:
    #     processed_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     logger.info("[Error] Failed to locate model response header")
    #     return None, solution_str
    processed_str = solution_str

    # Extract final answer using XML-style tags
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    
    if not matches:
        logger.info("[Error] No valid answer tags found")
        return processed_str.replace("<｜end▁of▁sentence｜>", "").strip(), processed_str
        
    final_answer = matches[-1].group(1).strip()
    return final_answer, processed_str


def parse_solution_text_format(solution_text: str) -> Dict[str, str]:
    """Parses ground truth solution text into status dictionary.
    
    Args:
        solution_text: Formatted solution text from dataset
        
    Returns:
        Dictionary mapping character names to their roles (knight/knave)
    """
    status_dict = {}
    logger.info("\n[Ground Truth Parsing]")
    
    for line in solution_text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        match = re.search(r'\b([A-Za-z]+)\b.*?\b(knight|knave)\b', line, re.IGNORECASE) # TODO: adapt it to multilingual
        if match:
            name, role = match.groups()
            status_dict[name] = role.lower()
            logger.info(f"  Found: {name} → {role}")
        else:
            logger.info(f"  [Warning] Unparseable line: '{line}'")
    
    return status_dict


def parse_solution_text_format_multilingual(solution_text: str, lang: str) -> Dict[str, str]:
    """
    Parses ground truth solution text into a status dictionary for multiple languages.

    Args:
        solution_text: Formatted solution text from dataset
        lang: Language code for role keyword mapping

    Returns:
        Dictionary mapping character names to their roles (knight/knave)
    """
    status_dict = {}
    logger.info("\n[Ground Truth Parsing]")

    # Normalize Unicode text (especially useful for Arabic)
    solution_text = unicodedata.normalize("NFKC", solution_text)

    # Multilingual persone names and role keywords
    name_translations = load_name_translations(lang)
    knight_keywords = ROLE_KEYWORDS.get(lang, {}).get("knight", [])
    knave_keywords = ROLE_KEYWORDS.get(lang, {}).get("knave", [])
    
    for line in solution_text.split('\n'):
        line = line.strip()
        line_lower = line.lower()
        if not line:
            continue

        found = False
        
        for keyword in knight_keywords:
            if keyword in line_lower:
                for name in name_translations:
                    name_lower = name.lower()
                    match = re.search(rf"({re.escape(name_lower)}).*?{re.escape(keyword)}", line_lower)
                    if match:
                        # name = match.group(1)
                        status_dict[name] = "knight"
                        logger.info(f"  Found: {name} → knight ({keyword})")
                        found = True
                        break

        if not found:
            for keyword in knave_keywords:
                if keyword in line_lower:
                    for name in name_translations:
                        name_lower = name.lower()
                        match = re.search(rf"({re.escape(name_lower)}).*?{re.escape(keyword)}", line_lower)
                        if match:
                            # name = match.group(1)
                            status_dict[name] = "knave"
                            logger.info(f"  Found: {name} → knave ({keyword})")
                            found = True
                            break

        if not found:
            logger.info(f"  [Warning] Unparseable line: '{line}'")

    return status_dict


def parse_model_answer(answer_text: str, expected_names: list) -> Optional[Dict[str, str]]:
    """Parses model's answer text into status dictionary.
    
    Args:
        answer_text: Text extracted from model's <answer> tags
        expected_names: List of character names requiring identification
        
    Returns:
        Dictionary mapping character names to predicted roles, or None if incomplete
    """
    status_dict = {}
    logger.info("\n[Model Answer Parsing]")
    logger.info(f"  Expected characters: {expected_names}")
    
    for name in expected_names:
        pattern = re.compile(
            rf'\b{re.escape(name)}\b.*?\b(knight|knave)\b',  # TODO: -> adapt to multilingual
            re.IGNORECASE
        )
        match = pattern.search(answer_text)
        
        if match:
            role = match.group(1).lower()
            status_dict[name] = role
            logger.info(f"  Found: {name} → {role}")
        else:
            logger.info(f"  [Error] Missing identification for {name}")
            return None
    
    return status_dict


def parse_model_answer_multilingual(answer_text: str, expected_names: list, lang) -> Optional[Dict[str, str]]:
    """Parses model's answer text into status dictionary.

    Args:
        answer_text: Text extracted from model's <answer> tags
        expected_names: List of character names requiring identification
        lang: Language code to use for role keyword mapping

    Returns:
        Dictionary mapping character names to predicted roles, or None if incomplete
    """
    status_dict = {}
    logger.info("\n[Model Answer Parsing]")
    logger.info(f"  Language: {lang}")
    logger.info(f"  Expected characters: {expected_names}")

    if lang not in ROLE_KEYWORDS:
        logger.info(f"  [Error] Unsupported language: {lang}")
        return None

    knight_keywords = ROLE_KEYWORDS[lang]["knight"]
    knave_keywords = ROLE_KEYWORDS[lang]["knave"]

    for name in expected_names:
        name_found = False

        for keyword in knight_keywords:
            pattern = re.compile(
                rf'{re.escape(name)}.*?({re.escape(keyword)})',
                re.IGNORECASE
            )
            if pattern.search(answer_text):
                status_dict[name] = "knight"
                logger.info(f"  Found: {name} → knight ({keyword})")
                name_found = True
                break

        if not name_found:
            for keyword in knave_keywords:
                pattern = re.compile(
                    rf'{re.escape(name)}.*?({re.escape(keyword)})',
                    re.IGNORECASE
                )
                if pattern.search(answer_text):
                    status_dict[name] = "knave"
                    logger.info(f"  Found: {name} → knave ({keyword})")
                    name_found = True
                    break

        if not name_found:
            logger.info(f"  [Error] Missing identification for {name}")
            return None

    return status_dict


def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    logger.info("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        # 'think_end': ('</think>', 1), # </think> is already removed
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        logger.info(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            logger.info(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if (
        # positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        logger.info("  [Error] Incorrect tag order: Expected ...</think><answer>...</answer>")
        validation_passed = False
    else:
        logger.info("  Tag sequence validation passed")

    return validation_passed


def parse_cot_eval_instruct(pred_str, lang, solution_text_format, verbose=False):
    logger.info("\n" + "="*80)
    logger.info(" Processing New Sample ".center(80, '='))
    
    # Parse ground truth data
    if lang == "en":
        gt_status = parse_solution_text_format(solution_text_format)
    else:
        gt_status = parse_solution_text_format_multilingual(solution_text_format, lang)
        
    expected_names = list(gt_status.keys())
    logger.info(f"[Ground Truth] Final identities: {gt_status}")

    # Extract model answer
    answer_text, processed_str = extract_solution(pred_str)
    logger.info(f"\n[Model Response]\n{processed_str}")

    # Validate response structure
    # format_correct = validate_response_structure(processed_str)
    # logger.info(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")

    # Validate answer content
    answer_score = 0
    is_correct = False
    correct_ratio = 0
    wrong_reason = "no_conclusion_matched"
    # if format_correct and answer_text:
    if answer_text:
        if lang == "en":
            pred_status = parse_model_answer(answer_text, expected_names)
        else:
            pred_status = parse_model_answer_multilingual(answer_text, expected_names, lang)
            
        if pred_status:
            logger.info(f"\n[Content Validation]")
            logger.info(f"  Expected: {gt_status}")
            logger.info(f"  Predicted: {pred_status}")
            
            if pred_status == gt_status:
                answer_score = 2
                is_correct = True
                correct_ratio = 1
                wrong_reason = None
                logger.info("  Content validation: FULL MATCH")
            else:
                answer_score = -1.5
                correct_ratio = 0
                wrong_reason = "wrong_identity"
                logger.info("  Content validation: MISMATCH")
        else:
            answer_score = -2
            correct_ratio = 0
            wrong_reason = "no_conclusion_matched"
            logger.info( "Fail to parse answer")
    else:
        print("\n[Content Validation] Skipped due to format errors or missing answer")
    
    if is_correct == False and verbose == True:
        logger.info("wrong_reason:",wrong_reason)
        logger.info("********* \nprediction before parse:\n", pred_str)
        logger.info("********* \nprediction after parse:\n", answer_text)

    return is_correct, answer_text, wrong_reason, correct_ratio


def batch_evaluate_file(file_path: str, lang: str):
    logger.info(f"📂 Processing: {file_path}")
    
    # Load data
    with open(file_path, "r", encoding="utf-8") as f:
        data_dict = json.load(f)

    modified = False
    for i, item in tqdm(enumerate(data_dict["data"])):
        solution = item.get("solution", [""])[-1]  # get latest model solution
        target = item.get("target", "")
        
        # # Skip if already annotated
        # if "is_correct" in item and "wrong_reason" in item:
        #     continue
        
        if solution is None:
            item["is_correct"] = None
            item["wrong_reason"] = "endless_reasoning"
            modified = True
        else:
            try:
                if isinstance(solution, list) and solution != []:
                    solution = solution[0]
                is_correct, _, wrong_reason, _ = parse_cot_eval_instruct(
                    pred_str=solution,
                    lang=lang,
                    solution_text_format=target
                )
                item["is_correct"] = is_correct
                item["wrong_reason"] = None if is_correct else wrong_reason
                modified = True
            except Exception as e:
                print(f"⚠️ Error on item {i}: {e}")
                item["is_correct"] = False
                item["wrong_reason"] = "parsing_error"
                modified = True

    # Save updated file
    if modified:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data_dict, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Updated file saved: {file_path}")
    else:
        logger.info(f"✅ No changes needed for: {file_path}")


if __name__ == "__main__":
    
    ####################################################################################################################
    #################################################### Text Cases ####################################################
    ####################################################################################################################
    
    lang = "ja"
    answer =  "<answer>\n(1) ゾーイは騎士です  \n(2) ジョセフは騎士です  \n(3) ソフィアは騎士です  \n</answer>"
    target = "(1) オーウェンはならず者です  \n(2) ジョセフはならず者です  \n(3) ソフィアはならず者です"
    
    lang = "ar"
    answer = "<answer>\n(1) مايكل هو الفارس\n(2) هنري هو المحتال\n(3) آفا قد تكون المحتال أو الفارس\n</answer>"
    target = "(1) مايكل فارس  \n(2) آفا فارس  \n(3) هنري فارس"
    
    lang = "zh"
    answer = "<answer>\n(1) 艾娃是骑士\n(2) 斯嘉丽是无赖\n(3) 本杰明是无赖\n</answer>"
    target = "(1) 斯嘉丽是无赖  \n(2) 艾娃是无赖  \n(3) 本杰明是无赖"
    
    lang = "fr"
    target = "(1) Émilie est une Chevalière  \n(2) Liam est un Chevalier  \n(3) Élisabeth est une Vaurienne"
    answer = "<answer>  \n(1) Émilie est un chevalier  \n(2) Liam est un chevalier  \n(3) Élisabeth est une vaurienne  \n</answer>"
    
    final_answer, processed_str = extract_solution(answer)
    status_dict_target = parse_solution_text_format_multilingual(target, lang)
    status_dict_answer = parse_model_answer_multilingual(answer, list(status_dict_target.keys()), lang)
    validation_passed = validate_response_structure(processed_str)
    
    is_correct, answer_text, wrong_reason, correct_ratio = parse_cot_eval_instruct(pred_str=answer, lang=lang, solution_text_format=target)
    
    
    answer = "(1) Ethan is a knight  \n(2) Abigail is a knight  \n(3) David is a knight  \n(4) Noah is a knave  \n\n<answer> (1) Ethan is a knight  \n(2) Abigail is a knight  \n(3) David is a knight  \n(4) Noah is a knave </answer>"
    target = "(1) Ethan is a knave\n(2) Abigail is a knight\n(3) David is a knight\n(4) Noah is a knight"
    is_correct, answer_text, wrong_reason, correct_ratio = parse_cot_eval_instruct(pred_str=answer, lang="en", solution_text_format=target)
    
    answer = "\n\n</answer>  \n<answer>  \n(1) अमेलिया एक शूरवीर है  \n(2) हार्पर एक धूर्त है  \n</answer><｜end▁of▁sentence｜>"
    answer =  "<answer>\n(1) एलिज़ाबेथ एक शूरवीर है  \n(2) स्कारलेट एक धूर्त है  \n(3) शार्लेट एक शूरवीर है  \n</answer>"
    target = "(1) एलिज़ाबेथ एक शूरवीर है  \n(2) स्कारलेट एक धूर्त है  \n(3) शार्लट एक शूरवीर है"
    is_correct, answer_text, wrong_reason, correct_ratio = parse_cot_eval_instruct(pred_str=answer, lang="hi", solution_text_format=target)
    ####################################################################################################################
    ################################################# Batch Evaluation #################################################
    ####################################################################################################################
    
    json_dir_list = [
        # "result_with_gt/DeepSeek-R1-Distill-Qwen-32B",
        # "result_with_gt/QwQ-32B",
        # "result_with_gt/DeepSeek-R1-Distill-Llama-70B",
        # "result_with_gt/DeepSeek-R1-Distill-Qwen-1.5B",
        # "result_with_gt/DeepSeek-R1-Distill-Qwen-7B",
        # "result_with_gt/DeepSeek-R1-Distill-Qwen-14B",
        # "result_with_gt/DeepSeek-R1-Distill-Llama-8B",
        # "result_reasoning_latin/DeepSeek-R1-Distill-Qwen-32B",
        # "result_reasoning_han/DeepSeek-R1-Distill-Qwen-32B",
        # "result_reasoning_input/DeepSeek-R1-Distill-Qwen-32B",
        # "result_reasoning_latin/QwQ-32B",
        # "result_reasoning_han/QwQ-32B",
        # "result_reasoning_input/QwQ-32B",
        # "result_reasoning_latin/DeepSeek-R1-Distill-Llama-70B",
        # "result_reasoning_han/DeepSeek-R1-Distill-Llama-70B",
        # "result_reasoning_input/DeepSeek-R1-Distill-Llama-70B",
        # "result_reasoning_latin/DeepSeek-R1-Distill-Llama-8B",
        # "result_reasoning_han/DeepSeek-R1-Distill-Llama-8B",
        # "result_reasoning_input/DeepSeek-R1-Distill-Llama-8B",
        # "result_reasoning_latin/DeepSeek-R1-Distill-Qwen-14B",
        # "result_reasoning_han/DeepSeek-R1-Distill-Qwen-14B",
        # "result_reasoning_input/DeepSeek-R1-Distill-Qwen-14B",
        # "result_reasoning_latin_han/DeepSeek-R1-Distill-Qwen-32B",
        # "result_reasoning_input_latin_han/DeepSeek-R1-Distill-Qwen-32B",
        # "result_reasoning_input_latin/DeepSeek-R1-Distill-Qwen-32B",
        # "result_reasoning_input_han/DeepSeek-R1-Distill-Qwen-32B",
        # "result_reasoning_latin_han/DeepSeek-R1-Distill-Llama-70B",
        # "result_reasoning_input_latin_han/DeepSeek-R1-Distill-Llama-70B",
        # "result_reasoning_input_latin/DeepSeek-R1-Distill-Llama-70B",
        # "result_reasoning_input_han/DeepSeek-R1-Distill-Llama-70B",
        # "result_with_gt/OpenR1-Llama-8B-SFT",
        # "result_with_gt/OpenR1-Qwen-7B-SFT",
        # "result_with_gt/Qwen3-4B",
        # "result_with_gt/Qwen3-30B-A3B",
        # "result_with_gt/Qwen3-32B",
        # "result_reasoning_invert_control_latin_han/DeepSeek-R1-Distill-Llama-8B",
        # "result_with_gt/Gemini-2-Flash-Thinking",
        "result_with_gt/DeepSeek-R1",
    ]
    
    for json_dir in json_dir_list:
        for file_path in tqdm(sorted(get_latest_files(json_dir))):
            if not file_path.startswith("kk"):
                continue
            lang = file_path.split('_')[1]
            if lang not in ROLE_KEYWORDS:
                lang = "en"
            file_path = os.path.join(json_dir, file_path)
            batch_evaluate_file(file_path, lang)
        
        model_name = os.path.basename(json_dir)
        print(f"Evalution on {model_name} finished ")
    