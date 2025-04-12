import mysql.connector
import statistics
import re

# DB 연결
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="0000",
    database="danawa_crawler_data"
)
cursor = conn.cursor()

# GPU 이름 정제 함수
def extract_gpu_info(text):
    pattern = re.search(
        r'(지포스|라데온)\s+'
        r'(GTX|GT|RTX|RX|XT|XTX)?\s*'
        r'(\d{3,5})'
        r'(?:\s*(Ti|SUPER|XT|XTX))?'
        r'(?:\s*(Ti|SUPER|XT|XTX))?'
        r'.*?(\d{1,2}GB)',
        text, re.IGNORECASE
    )
    if pattern:
        brand, prefix, number, suffix1, suffix2, memory = pattern.groups()
        suffixes = []
        for s in (suffix1, suffix2):
            if s and s.upper() not in [x.upper() for x in suffixes]:
                suffixes.append(s.upper())
        model = f"{brand} {prefix or ''}{number}"
        if suffixes:
            model += ' ' + ' '.join(suffixes)
        model += f" {memory}"
        return model.strip()
    return None

# 1. GPU만 정제해서 임시 저장
cursor.execute("SELECT name, date, price FROM vga_ref")
rows = cursor.fetchall()

# {(refined_name, date): [price1, price2, ...]} 형태로 그룹핑
grouped = {}
for name, date, price in rows:
    refined = extract_gpu_info(name)
    if refined and price > 0:  # 가격이 0 초과인 경우만 포함
        key = (refined, date)
        grouped.setdefault(key, []).append(price)

# 2. 통계 계산 및 결과 저장
# 테이블은 다음 구조로 생성되어 있어야 함:
# CREATE TABLE ref_vga_stats (
#   num INT AUTO_INCREMENT PRIMARY KEY,
#   name VARCHAR(255),
#   date DATE,
#   avg_price INT,
#   min_price INT,
#   max_price INT,
#   std_dev FLOAT
# );

for (refined_name, date), price_list in grouped.items():
    avg_price = round(sum(price_list) / len(price_list))
    min_price = min(price_list)
    max_price = max(price_list)
    std_dev = round(statistics.stdev(price_list), 2) if len(price_list) > 1 else 0.0

    cursor.execute(
        "INSERT INTO ref_vga_stats (name, date, avg_price, min_price, max_price, std_dev) "
        "VALUES (%s, %s, %s, %s, %s, %s)",
        (refined_name, date, avg_price, min_price, max_price, std_dev)
    )

# 마무리
conn.commit()
cursor.close()
conn.close()
