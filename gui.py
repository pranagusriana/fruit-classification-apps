import tkinter as tk
import tkinter.font as tkFont
from tkinter import filedialog as fd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import joblib
from utils import *

# Load deep learning and non deep learning model
dl_model = tf.keras.models.load_model("deep_learning_model.h5")
ndl_model = joblib.load("non_deep_learning_model.pickle")

class App:
    def __init__(self, root):
        #setting title
        root.title("Fruit Classification Apps")
        
        #setting window size
        width = 800
        height = 600
        screenwidth = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        root.geometry(alignstr)
        root.resizable(width=False, height=False)

        # Define Browse Image Button
        self.BrowseImage_Btn = tk.Button(root)
        self.BrowseImage_Btn["bg"] = "#f0f0f0"
        ft = tkFont.Font(family='Times', size=10)
        self.BrowseImage_Btn["font"] = ft
        self.BrowseImage_Btn["fg"] = "#000000"
        self.BrowseImage_Btn["justify"] = "center"
        self.BrowseImage_Btn["text"] = "Browse Image"
        self.BrowseImage_Btn.place(x=340, y=50, width=120, height=30)
        self.BrowseImage_Btn["command"] = self.BrowseImage_Btn_command

        # Define Label for Prediction Non Deep Learning Method
        self.PredNDL_Label = tk.Label(root)
        ft = tkFont.Font(family='Times', size=10)
        self.PredNDL_Label["font"] = ft
        self.PredNDL_Label["fg"] = "#000000"
        self.PredNDL_Label["justify"] = "center"
        self.PredNDL_Label["text"] = "Prediction Fruit using Non-Deep Learning: ..."
        self.PredNDL_Label.place(x=230, y=530, width=340, height=30)

        # Define Label for Prediction Deep Learning Method
        self.PredDL_Label = tk.Label(root)
        self.ft = tkFont.Font(family='Times', size=10)
        self.PredDL_Label["font"] = ft
        self.PredDL_Label["fg"] = "#000000"
        self.PredDL_Label["justify"] = "center"
        self.PredDL_Label["text"] = "Prediction Fruit using Deep Learning: ..."
        self.PredDL_Label.place(x=230, y=480, width=340, height=30)

        # Define Frame to Show Graph
        self.Fig_Frm = tk.Frame(root)
        self.Fig_Frm.place(x=20, y=120, width=760, height=310)

    # Method for update prediction result for non deep learning method
    def update_pred_ndl(self, pred):
        self.PredNDL_Label["text"] = f"Prediction Fruit using Non-Deep Learning: {pred}"
    
    # Method for update prediction result for deep learning method
    def update_pred_dl(self, pred):
        self.PredDL_Label["text"] = f"Prediction Fruit using Deep Learning: {pred}"

    # Method for clear frame
    def clearFrame(self):
        # destroy all widgets from frame
        for widget in self.Fig_Frm.winfo_children():
            widget.destroy()
        
        # this will clear frame and frame will be empty
        # if you want to hide the empty panel then
        self.Fig_Frm.pack_forget()

    # Method command for Browse Image Button
    def BrowseImage_Btn_command(self):
        filetypes = (
            ('jpeg files', '*.jpg'),
            ('All files', '*.*')
        )

        filename = fd.askopenfilename(
            title='Select an image',
            initialdir='./',
            filetypes=filetypes
        )

        img = load_image(filename, img_size=224)
        prep_img = preprocess_image(img)
        freq = get_freq(prep_img)

        fig, axes = plt.subplots(1, 3, dpi=100)
        img_ax = axes[0]
        img_ax.imshow(img)
        img_ax.set_axis_off()
        img_ax.set_title("Image")

        prep_img_ax = axes[1]
        prep_img_ax.imshow(prep_img)
        prep_img_ax.set_axis_off()
        prep_img_ax.set_title("Preprocessed Image")

        freq_ax = axes[2]
        freq_ax.plot(freq)
        freq_ax.set_title("Hist Preprocessed Image")
        plt.tight_layout()

        self.clearFrame()
        pred_ndl = mapping_idx[ndl_model.predict([img])[0]]
        pred_dl = mapping_idx[np.argmax(dl_model.predict(np.array([img])), axis=1)[0]]

        canvas = FigureCanvasTkAgg(fig, master=self.Fig_Frm)  # A tk.DrawingArea.
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.update_pred_ndl(pred_ndl)
        self.update_pred_dl(pred_dl)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()

# Prana Gusriana