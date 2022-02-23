import numpy
import tkinter
import matplotlib

matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def re_draw(tol_s, tol_n):
    re_draw.f.clf()  # clear the figure
    re_draw.a = re_draw.f.add_subplot(111)
    if chkBtnVar.get():
        if tol_n < 2:
            tol_n = 2
        my_tree = create_tree(re_draw.rawDat, model_leaf, model_err, (tol_s, tol_n))
        y_hat = create_fore_cast(my_tree, re_draw.testDat, model_tree_eval)
    else:
        my_tree = create_tree(re_draw.rawDat, ops=(tol_s, tol_n))
        y_hat = create_fore_cast(my_tree, re_draw.testDat)
    re_draw.a.scatter(re_draw.rawDat[:, 0].flatten().A[0], re_draw.rawDat[:, 1].flatten().A[0], s=2)  # use scatter for data set
    re_draw.a.plot(re_draw.testDat, y_hat, linewidth=1.0)  # use plot for y_hat
    re_draw.canvas.draw()


def get_inputs():
    try:
        tol_n = int(tolNentry.get())
    except:
        tol_n = 10
        print("tol_n的值应为整数")
        tolNentry.delete(0, tkinter.END)
        tolNentry.insert(0, '10')
    try:
        tol_s = float(tolSentry.get())
    except:
        tol_s = 1.0
        print("tol_s的值应为小数")
        tolSentry.delete(0, tkinter.END)
        tolSentry.insert(0, '1.0')
    return tol_n, tol_s


def draw_new_tree():
    tol_n, tol_s = get_inputs()  # get values from Entry boxes
    re_draw(tol_s, tol_n)


root = tkinter.Tk()

re_draw.f = Figure(figsize=(5, 4), dpi=100)  # create canvas
re_draw.canvas = FigureCanvasTkAgg(re_draw.f, master=root)
re_draw.canvas.draw()
re_draw.canvas.get_tk_widget().grid(row=0, columnspan=3)

tkinter.Label(root, text="tolN").grid(row=1, column=0)
tolNentry = tkinter.Entry(root)
tolNentry.grid(row=1, column=1)
tolNentry.insert(0, '10')
tkinter.Label(root, text="tolS").grid(row=2, column=0)
tolSentry = tkinter.Entry(root)
tolSentry.grid(row=2, column=1)
tolSentry.insert(0, '1.0')
tkinter.Button(root, text="ReDraw", command=draw_new_tree).grid(row=1, column=2, rowspan=3)
chkBtnVar = tkinter.IntVar()
chkBtn = tkinter.Checkbutton(root, text="Model Tree", variable=chkBtnVar)
chkBtn.grid(row=3, column=0, columnspan=2)

re_draw.rawDat = numpy.mat(load_data_set(sourceFile))
re_draw.testDat = numpy.arange(min(re_draw.rawDat[:, 0]), max(re_draw.rawDat[:, 0]), 0.01)
re_draw(1.0, 10)

root.mainloop()
