import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import os
import pickle
import random
import librosa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


model = tf.keras.models.load_model("InSAES_model.h5")


# Extracting features from the Audio file
def extract_features(audio_file_path):

    audio, sr = librosa.load(audio_file_path, duration=3)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    return mfcc_mean.reshape(1, 13, 1, 1)


# Emotion analysis
def analyse_emotion(audio_file_path):

    features = extract_features(audio_file_path)
    predict = model.predict(features)

    emotions = ["Angry", "Disgusted", "Fear", "Happy", "Neutral", "Sad"]

    return emotions, predict[0] * 100


# Visualizing waveform
def plot_waveform(audio_file_path, y):

    audio, sr = librosa.load(audio_file_path, duration=3)

    y.clear()
    y.plot(audio, color="#30546D")
    y.axis("off")


class InSAES:
    def __init__(self, in_saes):
        self.in_saes = in_saes
        self.in_saes.title("InSAES")
        self.in_saes.geometry("800x700")
        self.show_lock_form()

    def show_lock_form(self):

        # Creating Lock screen
        self.lock_frame = ttk.Frame(self.in_saes, padding="10 10 10 10")
        self.lock_frame.pack(fill="both", expand=True)

        # Logo in Lock screen
        self.logo = tk.PhotoImage(file="logo.png")
        logo_label = ttk.Label(self.lock_frame, image=self.logo)
        logo_label.pack(pady=20)

        # Title in LogIn screen
        ttk.Label(self.lock_frame, text="Intelligent System for Analysing Emotional States",
                  font=("Times New Roman", 22, "bold")).pack(pady=20)

        # Password field
        ttk.Label(self.lock_frame, text="Please enter your unique ID password:",
                  font=("Times New Roman", 15, "italic")).pack(pady=(100, 10))

        self.password = ttk.Entry(self.lock_frame, font=("Times New Roman", 12), show="*")
        self.password.pack(pady=5)

        # Entry button
        ttk.Button(self.lock_frame, text="LogIn", command=self.check_login).pack(pady=30)

    def check_login(self):

        password = self.password.get()

        if password == "insaes_2001261038":
            self.lock_frame.destroy()
            self.show_main_content()
        else:
            messagebox.showerror("Error", "Wrong ID password!")

    def show_main_content(self):

        # Styling
        self.style = ttk.Style()
        self.style.theme_use("clam")

        self.style = ttk.Style()
        self.style.configure('TButton', font=('Arial', 12), padding=5)
        self.style.configure('TLabel', font=('Arial', 12), padding=5)

        # Main field
        main_frame = ttk.Frame(self.in_saes, padding="8 5 5 5")
        main_frame.pack(fill="both", expand=True)

        # Menu
        menu = ttk.Frame(main_frame, relief="sunken")
        menu.pack(side="left", fill="y", padx=(0, 5), pady=10)

        # Logo
        self.logo = tk.PhotoImage(file="logo.png")
        logo_label = ttk.Label(menu, image=self.logo)
        logo_label.pack(pady=10)

        # Add audio button
        add_button = ttk.Button(menu, text="Add audio", command=self.add_audio_file)
        add_button.pack(pady=10)

        # History
        clear_button = ttk.Button(menu, text="Clear", command=self.clear_history)
        clear_button.pack(pady=10)

        history = ttk.Label(menu, text="History:", font=("Arial", 12, "bold", "italic"))
        history.pack(pady=(0, 10))

        self.history_listbox = tk.Listbox(menu, bg="lightgrey", font=("Arial", 12), height=10, width=17)
        self.history_listbox.pack(fill='both', padx=10, pady=(0, 10))
        self.history_listbox.bind('<Double-1>', self.on_history_select)

        # Instructions field
        instructions = ttk.Label(menu, text="Instructions:", font=("Arial", 12, "bold", "italic"))
        instructions.pack(pady=(15, 0))

        instructions_text = (
            '1. Click "Add audio" to choose audio fail.\n'
            '2. Wait for the analysis to finish.\n'
            '3. The results will be shown on the right of this field.\n'
            '4. Click "Clear" to clear the history.'
        )

        instructions_label = ttk.Label(menu, text=instructions_text, font=("Arial", 11), wraplength=120, justify="left")
        instructions_label.pack()

        # Result field
        result_frame = ttk.Frame(main_frame)
        result_frame.pack(side="left", fill="both", expand=True)

        # Slide bars
        self.slide_bars = ttk.Frame(result_frame)
        self.slide_bars.pack(pady=20)

        self.waveform = ttk.Frame(result_frame)
        self.waveform.pack(pady=20)

        # Saving history
        self.save_history()

    def add_audio_file(self):

        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])

        if file_path:
            self.show_loading()
            delay = random.randint(2000, 9000)
            self.in_saes.after(delay, lambda: self.process_audio_file(file_path))

    def show_loading(self):

        # Clear
        [x.destroy() for x in self.slide_bars.winfo_children()]
        [x.destroy() for x in self.waveform.winfo_children()]

        loading = ttk.Frame(self.slide_bars)
        loading.pack(expand=True)

        loading_label = ttk.Label(loading, text="Scanning...", font=("Arial", 16))
        loading_label.pack(pady=(250, 20))

        icon = ttk.Label(loading, text="⏳", font=("Arial", 30))
        icon.pack(pady=10)

    def process_audio_file(self, audio_file_path):
        try:
            x, y = analyse_emotion(audio_file_path)

            self.show_progress_bars(x, y)

            self.animate_slide_bars(y)

            self.show_waveform(audio_file_path)

            self.update_history(audio_file_path)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_progress_bars(self, x, y):
        # Clear
        [x.destroy() for x in self.slide_bars.winfo_children()]

        self.slide_t_bars = []

        ttk.Label(self.slide_bars, text="Result of the analysis",
                  font=("Ariel", 16, "underline")).pack(pady=(20, 10))

        for i in x:
            bar = ttk.Frame(self.slide_bars)
            bar.pack(fill="x", pady=10)

            label = ttk.Label(bar, text=i, font=("Arial", 12), width=10)
            label.pack(side="left")

            slide = ttk.Progressbar(bar, length=400, maximum=100)
            slide.pack(side="left", padx=20)

            self.slide_t_bars.append(slide)

    def animate_slide_bars(self, lst):

        def update_bars(step=0):
            if step <= 100:
                for i, k in enumerate(self.slide_t_bars):
                    k["value"] = (lst[i] / 100) * step

                self.in_saes.after(20, update_bars, step + 1)

        update_bars()

    def show_waveform(self, audio_file_path):
        # Clear
        [x.destroy() for x in self.waveform.winfo_children()]

        ttk.Label(self.waveform, text="Analysis of audio waveform", font=("Ariel", 16, "underline")).pack(pady=(0, 5))

        # Creating new waveform
        figure, ax = plt.subplots(figsize=(5, 2))
        plot_waveform(audio_file_path, ax)

        canvas = FigureCanvasTkAgg(figure, master=self.waveform)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def update_history(self, audio_file_path):

        if audio_file_path not in self.history:
            self.history.append(audio_file_path)

            if len(self.history) > 10:
                self.history.pop(0)

        self.history_listbox.delete(0, tk.END)

        for i in self.history:
            file_name = os.path.basename(i)
            self.history_listbox.insert(tk.END, file_name)
        self.saving_the_history()

    def clear_history(self):
        self.history = []
        self.history_listbox.delete(0, tk.END)
        self.saving_the_history()

    def saving_the_history(self):
        with open("InSAES_history.pkl", "wb") as f:
            pickle.dump(self.history, f)

    def save_history(self):
        if os.path.exists("InSAES_history.pkl"):
            with open("InSAES_history.pkl", "rb") as f:
                self.history = pickle.load(f)

                for i in self.history:
                    file_name = os.path.basename(i)
                    self.history_listbox.insert(tk.END, file_name)

        else:
            self.history = []

    def on_history_select(self, event):

        selection = event.widget.curselection()
        
        if selection:
            index = selection[0]
            audio_file_path = self.history[index]
            self.process_audio_file(audio_file_path)


if __name__ == "__main__":
    in_saes_main = tk.Tk()
    app = InSAES(in_saes_main)
    in_saes_main.mainloop()


# password: insaes_2001261038
