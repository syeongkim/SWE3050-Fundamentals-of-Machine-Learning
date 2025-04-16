import pandas as pd

# 파일 경로 설정
file1 = "submission_0415last.csv"
file2 = "submission_0415threshold0.5.csv"

# 1. CSV 파일 불러오기
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# 2. shape 다르면 바로 종료
if df1.shape != df2.shape:
    print(f"❌ shape이 다릅니다. df1: {df1.shape}, df2: {df2.shape}")
else:
    # 3. 컬럼 순서와 행 순서 정렬
    df1_sorted = df1.sort_values(by=df1.columns.tolist()).reset_index(drop=True)
    df2_sorted = df2.sort_values(by=df2.columns.tolist()).reset_index(drop=True)

    # 4. 값이 다른 셀 비교
    comparison = df1_sorted != df2_sorted
    if not comparison.any().any():
        print("✅ 두 파일의 내용이 완전히 같습니다!")
    else:
        print("❌ 차이 있는 셀 위치와 값:")
        diffs = []
        for row in range(comparison.shape[0]):
            for col in comparison.columns[comparison.iloc[row]]:
                diffs.append({
                    "Row": row,
                    "Column": col,
                    "File1": df1_sorted.loc[row, col],
                    "File2": df2_sorted.loc[row, col]
                })
        diff_df = pd.DataFrame(diffs)
        print(diff_df)