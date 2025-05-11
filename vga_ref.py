import mysql.connector
import statistics
import re
import numpy as np
from sklearn.cluster import KMeans
from db_config import get_connection
import time
from collections import defaultdict
from scipy.stats import zscore

def extract_gpu_info(text):
    pattern = re.search(
        r'(지포스|라데온)\s+'  # 브랜드
        r'(GTX|RTX|RX)?\s*'  # 시리즈
        r'(\d{3,5})'  # 숫자 (모델 번호)
        r'(?:\s*(Ti|SUPER|XT|XTX))?'  # 옵션 (예: Ti, SUPER 등)
        r'(?:\s*(Ti|SUPER|XT|XTX))?'  # 옵션 (예: Ti, SUPER 등)
        r'.*?(\d{1,2}GB)',  # 메모리 크기
        text, re.IGNORECASE
    )
    if pattern:
        brand, prefix, number, suffix1, suffix2, memory = pattern.groups()
        suffixes = {s.upper() for s in (suffix1, suffix2) if s}
        model = f"{brand} {prefix or ''}{number}"
        if suffixes:
            model += ' ' + ' '.join(suffixes)
        model += f" {memory}"
        return model.strip()
    return None

def remove_outliers_with_zscore(prices, threshold=2):
    """ Z-score를 이용해 이상치 제거 """
    if len(prices) < 2:
        return prices  # 가격 데이터가 2개 미만일 경우 그대로 반환

    # 분산이 너무 작으면 Z-score 계산을 건너뛰고 원본 데이터를 반환
    if np.var(prices) < 1e-6:
        return prices

    z_scores = zscore(prices)
    return [price for price, z in zip(prices, z_scores) if abs(z) <= threshold]

def run():
    start_time = time.time()
    conn = get_connection()
    cursor = conn.cursor()

    # 1. GPU만 정제해서 가져오기
    cursor.execute("SELECT name, date, price FROM vga_ref")
    rows = cursor.fetchall()

    grouped = defaultdict(list)
    print("데이터 로딩 완료")  # 데이터 로딩 완료 확인
    for name, date, price in rows:
        refined = extract_gpu_info(name)
        if refined and price > 0:
            key = (refined, date)
            grouped.setdefault(key, []).append(price)

    print(f"총 {len(grouped)}개의 GPU 데이터가 그룹화되었습니다.")  # 그룹화된 데이터 수 확인

    # 2. 통계 계산 및 저장
    bulk_insert_data = []
    batch_size = 10000

    for (refined_name, date), price_list in grouped.items():
        if len(price_list) < 2:
            continue

        print(f"클러스터링 중: {refined_name} ({date})")  # 클러스터링 진행 중 확인

        # 이상치 제거
        cleaned_prices = remove_outliers_with_zscore(price_list)

        # 이상치 제거 후 가격이 없으면 건너뜁니다.
        if len(cleaned_prices) < 1:
            print(f"이상치 제거 후 가격 데이터가 부족하여 건너뜁니다: {refined_name} ({date})")
            continue

        # 클러스터링이 가능한 시점부터 통합 처리
        prices_np = np.array(cleaned_prices).reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(prices_np)

        # 그룹별 평균 계산
        cluster_data = {}
        for label in set(labels):
            cluster_prices = [price for price, l in zip(cleaned_prices, labels) if l == label]
            cluster_data[label] = {
                'prices': cluster_prices,
                'avg': np.mean(cluster_prices)
            }

        # 평균 기준으로 정렬
        sorted_labels = sorted(cluster_data.items(), key=lambda x: x[1]['avg'])

        # 보급/상급 등급 이름
        tiers = ["보급형", "상급형"]

        # 실제 클러스터 수에 맞게 매핑
        label_to_tier = {label: tiers[i] for i, (label, _) in enumerate(sorted_labels)}

        # 저장
        for label, info in cluster_data.items():
            prices = info['prices']
            tier = label_to_tier[label]
            avg_price = round(np.mean(prices))
            min_price = min(prices)
            max_price = max(prices)
            std_dev = round(statistics.stdev(prices), 2) if len(prices) > 1 else 0.0

            refined_with_tier = f"{refined_name} ({tier})"
            bulk_insert_data.append((refined_with_tier, date, avg_price, min_price, max_price, std_dev))

    # 한 번에 데이터를 삽입
    if bulk_insert_data:
        print(f"{len(bulk_insert_data)}개의 데이터가 삽입될 준비가 되었습니다.")  # 데이터 삽입 준비 상태 확인
        cursor.executemany(
            "INSERT INTO ref_vga_stats (name, date, avg_price, min_price, max_price, std_dev) "
            "VALUES (%s, %s, %s, %s, %s, %s)", bulk_insert_data
        )

    conn.commit()
    cursor.close()
    conn.close()
    end_time = time.time()
    elapsed_time = round(end_time - start_time, 2)
    print(f"2단계 분류 통계 완료: ref_vga_stats 테이블에 저장됨. 소요 시간: {elapsed_time}초")

if __name__ == "__main__":
    run()
