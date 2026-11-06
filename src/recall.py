import os
import sys
import json


def recall(student_search_results_path: str,
           dataset_path: str) -> None:
    """
    Get the recall@k by comparing the search results to the groud test
    """
    if (not os.path.exists(student_search_results_path)
            or not os.path.exists(dataset_path)):
        print("Error: One of the provided files does not exist.",
              file=sys.stderr)
        sys.exit(1)

    with open(dataset_path, "r", encoding="UTF-8") as f:
        ground_truth_data = json.load(f)

    with open(student_search_results_path, "r", encoding="UTF-8") as f:
        student_data = json.load(f)

    gt_questions = ground_truth_data.get("rag_questions", ground_truth_data)
    gt_dict = {}
    for q in gt_questions:
        if "sources" in q and q["sources"]:
            gt_dict[q["question_id"]] = q["sources"]

    search_results = student_data.get("search_results", [])
    total_questions = 0
    k_values = [1, 3, 5, 10]
    recalls = {k: 0 for k in k_values if k <= student_data.get("k", 5)}

    for result in search_results:
        q_id = result["question_id"]

        if q_id not in gt_dict:
            continue

        total_questions += 1
        expected_sources = gt_dict[q_id]
        retrieved_sources = result.get("retrieved_sources", [])

        for k in recalls.keys():
            top_k_retrieved = retrieved_sources[:k]

            found = False
            for expected in expected_sources:
                for retrieved in top_k_retrieved:
                    if expected["file_path"] == retrieved["file_path"]:
                        exp_start = expected["first_character_index"]
                        exp_end = expected["last_character_index"]
                        ret_start = retrieved["first_character_index"]
                        ret_end = retrieved["last_character_index"]
                        if max(exp_start, ret_start) <= min(exp_end, ret_end):
                            found = True
                            break
                if found:
                    break
            if found:
                recalls[k] += 1
    if total_questions == 0:
        print("Aucune question commune trouvée entre les deux fichiers.\
Vérifiez les question_id.")
        return
    print(f"\n{'='*40}")
    print(f" ÉVALUATION DU RETRIEVAL (Total: {total_questions} questions)")
    print(f"{'='*40}")

    for k in sorted(recalls.keys()):
        recall_score = (recalls[k] / total_questions) * 100
        print(f" Recall@{k} : {recall_score:.2f}%\
({recalls[k]}/{total_questions})")
    print(f"{'='*40}\n")
