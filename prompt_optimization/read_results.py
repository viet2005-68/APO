"""
Script để đọc và hiển thị kết quả evaluation từ file output
"""

import json
import ast
import sys
import os

def parse_output_file(filepath):
    """Parse file output và trả về structured data"""
    if not os.path.exists(filepath):
        print(f"File không tồn tại: {filepath}")
        return None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if not lines:
        print("File rỗng")
        return None
    
    # Parse config (dòng đầu tiên)
    try:
        config = json.loads(lines[0].strip())
    except:
        config = None
    
    rounds = []
    current_round = None
    
    i = 1
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith("======== ROUND"):
            # Lưu round trước đó nếu có
            if current_round is not None:
                rounds.append(current_round)
            
            # Parse round number
            round_num = int(line.split("ROUND")[1].strip())
            current_round = {
                "round": round_num,
                "time": None,
                "candidates": None,
                "estimated_scores": None,
                "f1_scores": None
            }
            
            # Dòng tiếp theo là thời gian
            if i + 1 < len(lines):
                try:
                    current_round["time"] = float(lines[i + 1].strip())
                except:
                    pass
                i += 1
            
            # Dòng tiếp theo là candidates
            if i + 1 < len(lines):
                try:
                    candidates_str = lines[i + 1].strip()
                    current_round["candidates"] = ast.literal_eval(candidates_str)
                except:
                    current_round["candidates"] = [candidates_str]
                i += 1
            
            # Dòng tiếp theo là estimated scores
            if i + 1 < len(lines):
                try:
                    scores_str = lines[i + 1].strip()
                    current_round["estimated_scores"] = ast.literal_eval(scores_str)
                except:
                    pass
                i += 1
            
            # Dòng tiếp theo là F1 scores
            if i + 1 < len(lines):
                try:
                    f1_str = lines[i + 1].strip()
                    current_round["f1_scores"] = ast.literal_eval(f1_str)
                except:
                    pass
                i += 1
        
        i += 1
    
    # Lưu round cuối cùng
    if current_round is not None:
        rounds.append(current_round)
    
    return {
        "config": config,
        "rounds": rounds
    }


def print_results(data, verbose=False):
    """In kết quả ra màn hình"""
    if data is None:
        return
    
    print("=" * 80)
    print("KẾT QUẢ EVALUATION")
    print("=" * 80)
    
    # In config
    if data["config"]:
        config = data["config"]
        print(f"\n📋 Cấu hình:")
        print(f"  - Task: {config.get('task', 'N/A')}")
        print(f"  - Data dir: {config.get('data_dir', 'N/A')}")
        print(f"  - Rounds: {config.get('rounds', 'N/A')}")
        print(f"  - Beam size: {config.get('beam_size', 'N/A')}")
        print(f"  - Test examples: {config.get('n_test_exs', 'N/A')}")
        print(f"  - Evaluator: {config.get('evaluator', 'N/A')}")
        print(f"  - Temperature: {config.get('temperature', 'N/A')}")
    
    # In từng round
    print(f"\n{'=' * 80}")
    print(f"📊 TỔNG QUAN: {len(data['rounds'])} rounds")
    print(f"{'=' * 80}")
    
    for round_data in data["rounds"]:
        round_num = round_data["round"]
        time_taken = round_data["time"]
        candidates = round_data["candidates"]
        est_scores = round_data["estimated_scores"]
        f1_scores = round_data["f1_scores"]
        
        print(f"\n🔹 ROUND {round_num}")
        print(f"   Thời gian: {time_taken:.4f} giây" if time_taken else "   Thời gian: N/A")
        
        if f1_scores:
            best_f1 = max(f1_scores)
            best_idx = f1_scores.index(best_f1)
            print(f"   📈 F1 Score tốt nhất: {best_f1:.4f}")
            print(f"   📊 Tất cả F1 Scores: {[f'{s:.4f}' for s in f1_scores]}")
        
        if est_scores:
            print(f"   🎯 Estimated Scores: {[f'{s:.4f}' for s in est_scores]}")
        
        if candidates and verbose:
            print(f"\n   📝 Prompts ({len(candidates)} candidates):")
            for idx, candidate in enumerate(candidates):
                print(f"\n   Candidate {idx + 1}:")
                # Chỉ in 200 ký tự đầu để không quá dài
                preview = candidate[:200].replace('\n', ' ')
                print(f"   {preview}{'...' if len(candidate) > 200 else ''}")
                if f1_scores and idx < len(f1_scores):
                    print(f"   F1 Score: {f1_scores[idx]:.4f}")
    
    # Tóm tắt
    if data["rounds"]:
        print(f"\n{'=' * 80}")
        print("📈 TÓM TẮT")
        print(f"{'=' * 80}")
        
        all_f1 = []
        for round_data in data["rounds"]:
            if round_data["f1_scores"]:
                all_f1.extend(round_data["f1_scores"])
        
        if all_f1:
            print(f"  - F1 Score cao nhất: {max(all_f1):.4f}")
            print(f"  - F1 Score thấp nhất: {min(all_f1):.4f}")
            print(f"  - F1 Score trung bình: {sum(all_f1)/len(all_f1):.4f}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python read_results.py <output_file> [--verbose]")
        print("\nVí dụ:")
        print("  python read_results.py evaluate_result.out")
        print("  python read_results.py evaluate_result.out --verbose")
        sys.exit(1)
    
    filepath = sys.argv[1]
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    
    data = parse_output_file(filepath)
    print_results(data, verbose=verbose)

