

-------------------------테이블생성------------------------

CREATE TABLE `danawa_crawler_data`.`vga_price` (
  `num` INT NOT NULL AUTO_INCREMENT,
  `name` VARCHAR(255) NULL,
  `date` DATE NULL,
  `price` INT NULL,
  PRIMARY KEY (`num`));


CREATE TABLE ref_vga_stats (
  num INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(255),
  date DATE,
  avg_price INT,
  min_price INT,
  max_price INT,
  std_dev FLOAT
);

-------------------------단종제거------------------------
연속된 날짜 구간에서 가격이 0인 데이터만 필터링
연속된 날짜들끼리 그룹핑하기 위해 ROW_NUMBER() 윈도우 함수 사용
그 그룹 중에서 날짜 수가 20 이상인 name만 추출
해당 name의 데이터를 전부 삭제


WITH zero_price_rows AS (
  SELECT
    name,
    date,
    price,
    ROW_NUMBER() OVER (PARTITION BY name ORDER BY date) AS rn
  FROM danawa_crawler_data.vga_price
  WHERE price = 0
),
grouped_zero AS (
  SELECT
    name,
    date,
    DATE_SUB(date, INTERVAL rn DAY) AS grp
  FROM zero_price_rows
),
grouped_count AS (
  SELECT
    name,
    COUNT(*) AS zero_days
  FROM grouped_zero
  GROUP BY name, grp
  HAVING COUNT(*) >= 20
)
DELETE FROM danawa_crawler_data.vga_price
WHERE name IN (SELECT name FROM grouped_count);

