import tkinter as tk
from tkinter import ttk
from model_utils import load_data, preprocess_data, train_model
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
import mplcursors
import matplotlib
import matplotlib.pyplot as plt
# 한글 폰트 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False


class PricePredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GPU 가격 예측기")
        self.root.geometry("1000x800")

        self.df = preprocess_data(load_data())
        self.gpu_list = sorted(self.df['name'].unique())

        self.create_widgets()

    def create_widgets(self):
        ttk.Label(self.root, text="그래픽카드 선택:").pack(pady=10)
        self.gpu_combo = ttk.Combobox(self.root, values=self.gpu_list, width=50)
        self.gpu_combo.pack()

        self.predict_btn = ttk.Button(self.root, text="2개월 예측하기", command=self.predict)
        self.predict_btn.pack(pady=20)

        self.result_label = ttk.Label(self.root, text="", font=("Helvetica", 14))
        self.result_label.pack(pady=10)

    def predict(self):
        gpu_name = self.gpu_combo.get()
        if not gpu_name:
            self.result_label.config(text="그래픽카드를 선택하세요.")
            return

        model, latest_row, rmse = train_model(self.df, gpu_name)
        if model is None:
            self.result_label.config(text="데이터가 충분하지 않습니다.")
            return

        # 2개월치 예측 (60일)
        pred_prices, future_dates = self.predict_next_n_days(model, latest_row, gpu_name, 60)

        # 과거 데이터
        df_gpu = self.df[self.df['name'] == gpu_name].copy()
        self.draw_plot(df_gpu, future_dates, pred_prices)

        self.result_label.config(
            text=f"{gpu_name}의 2개월 예측 완료 (RMSE: {int(rmse):,})"
        )

    def predict_next_n_days(self, model, latest_row, gpu_name, days=60):
        preds = []
        dates = []

        row = latest_row.copy()

        df_gpu = self.df[self.df['name'] == gpu_name]
        current_date = df_gpu['date'].max()

        future_dates = pd.bdate_range(start=current_date + pd.Timedelta(days=1), periods=days)

        for day in future_dates:
            log_pred = model.predict(row)[0]
            pred_price = np.expm1(log_pred)

            preds.append(pred_price)
            dates.append(day)

            row = row.copy()
            row.loc[row.index[0], 'avg_price'] = log_pred

        last_real_price = np.expm1(latest_row['avg_price'].values[0])
        offset = last_real_price - preds[0]
        preds = [p + offset for p in preds]

        return preds, dates

    def draw_plot(self, df_gpu, future_dates, pred_prices):
        plt.close('all')

        fig, ax = plt.subplots(figsize=(10, 4), dpi=100)

        df_gpu['avg_price_real'] = np.expm1(df_gpu['avg_price'])

        # 과거 가격 (실제 가격)
        ax.plot(df_gpu['date'], df_gpu['avg_price_real'], label='과거 평균가', color='blue')

        # 예측 가격 (미래 가격)
        ax.plot(future_dates, pred_prices, label='예측가 (60일)', color='red', linestyle='--')

        # 예측 시작점
        ax.plot(future_dates[:1], pred_prices[:1], 'o', color='green', label='예측 시작점')

        # 과거 데이터와 예측 구분 선
        ax.axvline(df_gpu['date'].max(), color='gray', linestyle='--', alpha=0.5)

        ax.set_title('가격 추세 및 2개월 예측')
        ax.set_xlabel('날짜')
        ax.set_ylabel('가격 (원)')
        ax.legend()

        fig.autofmt_xdate(rotation=30)
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))

        if hasattr(self, 'canvas'):
            self.canvas.get_tk_widget().destroy()

        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(pady=10)

        plt.close(fig)

        def format_tooltip(date, price):
            return f"날짜: {date.strftime('%Y-%m-%d')}\n가격: {int(price):,}원"

        cursor1 = mplcursors.cursor(ax.lines[0], hover=True)

        @cursor1.connect("add")
        def on_add(sel):
            index = int(round(sel.index))
            date = df_gpu['date'].iloc[index]
            price = df_gpu['avg_price_real'].iloc[index]
            sel.annotation.set(text=format_tooltip(date, price))

        cursor2 = mplcursors.cursor(ax.lines[1], hover=True)

        @cursor2.connect("add")
        def on_add(sel):
            index = int(round(sel.index))
            date = future_dates[index]
            price = pred_prices[index]
            sel.annotation.set(text=format_tooltip(date, price))


if __name__ == "__main__":
    root = tk.Tk()
    app = PricePredictorApp(root)
    root.mainloop()
