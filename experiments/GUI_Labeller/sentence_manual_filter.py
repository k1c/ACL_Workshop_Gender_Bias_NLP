
# Gui to classify the sentences to A1 or otherwise from Ali Emami

import json
import tkinter as tk
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('filename', help='path to the file to use')

args = parser.parse_args()
data = open(args.filename, 'r')
data=data.readlines()
nb_data = len(data)
print("nb_data:", nb_data)

data_with_annotations = []
try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

try:
    # else load stuff from saved file
    saved_data = json.load(open('labeled_sentences.json', 'r'))
    index = len(saved_data)
    data_with_annotations.extend(saved_data)
except FileNotFoundError:
    # if starting fresh
    index = 0


def counter_label(label):
    def count():
        label.config(text=data[index])
        label.after(1000, count)
    count()



def yes_on_click(event=None):
    global index
    new_data={}
    new_data['sentence']=data[index]
    new_data['is_A1'] = True
    new_data=json.dumps(new_data)
    data_with_annotations.append(new_data)
    outfile = open('labeled_sentences.json', 'w')
    outfile.write(json.dumps(data_with_annotations))
    print(index)
    index+=1
    if index > nb_data - 1:
        button_yes.config(state=tk.DISABLED)
        button_no.config(state=tk.DISABLED)
    else:
        button_yes.config(state=tk.NORMAL)
        button_no.config(state=tk.NORMAL)

def no_on_click(event=None):
    global index
    new_data={}
    new_data['sentence']=data[index]
    new_data['is_A1'] = False
    new_data=json.dumps(new_data)
    data_with_annotations.append(new_data)
    outfile = open('labeled_sentences.json', 'w')
    outfile.write(json.dumps(data_with_annotations))
    print(index)
    index += 1
    if index > nb_data - 1:
        button_yes.config(state=tk.DISABLED)
        button_no.config(state=tk.DISABLED)
    else:
        button_yes.config(state=tk.NORMAL)
        button_no.config(state=tk.NORMAL)


root = tk.Tk()
frame=tk.Frame(root, width=2000, height=250, bg="gray")
root.title("Sentence to check")
sentence_displayer = tk.Label(root, fg="black", font=("Helvetica", 16), width=120, height=5,
                              wraplength=900, justify=tk.LEFT)
sentence_displayer.pack()
counter_label(sentence_displayer)
candidate_displayer = tk.Label(root, fg="black",font=("Helvetica", 16), width=80, height=5,wraplength=900, justify=tk.CENTER)
candidate_displayer.pack()
button_yes = tk.Button(root, text="A1 Sentence", width=40, command=yes_on_click,
                            fg="dark green", bg = "white")
button_yes.pack()

button_no = tk.Button(root, text ="Other", width=40,command=no_on_click,
                            fg="dark red", bg = "white")
button_no.pack()


if index > nb_data - 1:
    button_yes.config(state=tk.DISABLED)
    button_bo.config(state=tk.DISABLED)
print(index)

root.mainloop()