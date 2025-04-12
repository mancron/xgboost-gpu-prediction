import os
import glob

# 'Data' 폴더 경로 설정
data_dir = "Data"

# Data 폴더 내 모든 월별 폴더 탐색
for month_folder in os.listdir(data_dir):
    month_path = os.path.join(data_dir, month_folder)

    if os.path.isdir(month_path):  # 월별 폴더가 맞는지 확인
        # 월별 폴더 내 모든 .csv 파일 찾기
        csv_files = glob.glob(os.path.join(month_path, "*.csv"))

        for file in csv_files:
            # 'VGA.csv' 파일은 삭제하지 않음
            if os.path.basename(file) != "VGA.csv":
                os.remove(file)  # 다른 csv 파일 삭제
                print(f"삭제된 파일: {file}")