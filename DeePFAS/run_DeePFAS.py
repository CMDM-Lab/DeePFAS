import tkinter as tk
import os
from tkinter import ttk, Button, IntVar, BooleanVar, Radiobutton, filedialog, font

import subprocess
from pyteomics import mgf
import h5py

def check_mgf_file():
    global spec_pth
    ok = False
    try:
        with mgf.read(spec_pth) as spec_f:
            for spec in spec_f:
                if spec['params'].get('mslevel') in [None, '']:
                    print('Spectrum data need to known mslevel\n')
                    
                elif spec['params'].get('title') in [None, '']:
                    print('Spectrum data need to known title\n')
                    
                elif spec['params'].get('precursor_type') in [None, '']:
                    print('Spectrum data need to known precursor_type\n')
                    
                elif not mode_var.get() and spec['params'].get('canonicalsmiles') in [None, '']:
                    print('Spectrum data need to known canonical smiles\n')
                    
                elif (str(spec['params']['mslevel']) not in ['MS2', '2']):
                    print('Only MS2 Spec is valid\n')
    
                elif (spec['params']['precursor_type'] not in ['[M-H]-', '[M-H]1-']):
                    print('Only precursor type [M-H]- is valid\n')

                else:
                    ok = True

    except:
        print(spec_pth)
        print('Error when loading MGF file\n')

    return ok

def check_ref_db():
    global db_pth

    with h5py.File(db_pth, 'r') as f:
        try:
            chunks_smiles = f['CANONICALSMILES'].chunks[0]
            chunks_chemical_emb = f['chemical_emb'].chunks[0]

        except ValueError:
            print('Error when loading HDF5 file\n')
            return False
    return True

def button_read_spectra_file():
    global ms2_file_pth, spec_pth

    spec_pth = filedialog.askopenfilename()
    if (spec_pth[-4:] != '.mgf'):
        ms2_file_pth.configure(text='Invalid file format')
        return
    ms2_file_pth.configure(text=os.path.basename(spec_pth)[:20])

def button_reference_seraching_database():
    global db_file_pth, db_pth

    db_pth = filedialog.askopenfilename()
    if (db_pth[-5:] != '.hdf5'):
        db_file_pth.configure(text='Invalid file format')
        return
    db_file_pth.configure(text=os.path.basename(db_pth)[:20])

def button_results_output():
    global results_file_pth, results_pth
    results_pth = filedialog.askdirectory()
    results_file_pth.configure(text=os.path.basename(results_pth)[:20])

def button_run_deepfas():
    global topk
    mode = 'inference' if mode_var.get() else 'eval'
    topk = topk_entry.get()
    try:
        topk = int(topk)
    except:
        print('topk need to be an integer\n')
        return

    if (topk < 1):
        print('topk can not less than one\n')
        return
    
    if (not check_mgf_file() or not check_ref_db()):
        return

    cmd = f"python3 DeePFAS/test_deepfas.py \
        --deepfas_config_pth DeePFAS/config/deepfas_config.json \
        --ae_config_pth ae/config/ae_config.json \
        --test_data_pth {spec_pth} \
        --retrieval_data_pth {db_pth} \
        --results_dir {results_pth} \
        --topk {int(topk)} \
        --mode {mode} \
    "

    subprocess.run(cmd, shell=True, **log_cmds)


topk = 20

root = tk.Tk(className='DeePFAS')
root.title('DeePFAS')
root.geometry('640x400')
root.resizable(width=False, height=False)

spec_pth = ''
db_pth = 'mol_dataset/5w_chemical_embbeddings.hdf5'
results_pth = './'
mode = 'inference'
log_cmds = {}

s = ttk.Style()
s.configure('Red.TLabelframe.Label', font=('Courier', 13, 'bold'))
label_frame = ttk.LabelFrame(root, text='DeePFAS', style = "Red.TLabelframe")
label_frame.place(x=30, y=10, width=580, height=380)

font_size = 16
button_font_size = 12
myfont = font.Font(size=button_font_size, weight='bold')

ms2_file_pth = Button(text="Load MS2 Spectra (.mgf)", bg='#E7ECF7', command=button_read_spectra_file)
ms2_file_pth.place(x=150, y=35, height=30, width=200)
ms2_file_pth['font'] = myfont

MS2_file_Title = ttk.Label(root,  text='MS2 file: ')
MS2_file_Title.place(x=40, y=40)
MS2_file_Title.configure(font=('Courier', font_size))

db_file_pth = Button(text="Load Molecule Database (.hdf5)", bg='#E7ECF7', command=button_reference_seraching_database)
db_file_pth.place(x=350, y=70, height=30, width=250)
db_file_pth['font'] = myfont

db_file_title = ttk.Label(root,  text='Searching Molecules DB file: ')
db_file_title.place(x=40, y=75)
db_file_title.configure(font=('Courier', font_size))

results_file_pth = Button(text="Load Output Results Dir", bg='#E7ECF7', command=button_results_output)
results_file_pth.place(x=300, y=105, height=30, width=200)
results_file_pth['font'] = myfont

results_file_title = ttk.Label(root,  text='Output Folder (Results): ')
results_file_title.place(x=40, y=110)
results_file_title.configure(font=('Courier', font_size))


topk_title = ttk.Label(root,  text='Topk Candidates: ')
topk_title.place(x=40, y=145)
topk_title.configure(font=('Courier', font_size))

topk_entry = ttk.Entry(root)
topk_entry.place(x=220, y=143, height=25, width=40)
topk_entry.insert('end', topk)

mode_title = ttk.Label(root, text='Mode :')
mode_title.place(x=40, y=180)
mode_title.configure(font=('Courier', font_size))

mode_var = BooleanVar()
inference_mode = Radiobutton(root, text="Inference", variable=mode_var, value=True)
inference_mode.place(x=120, y=178)
eval_mode = Radiobutton(root, text="Eval", variable=mode_var, value=False)
eval_mode.place(x=240, y=178)
mode_var.set(True)

execute_button = Button(text="Run", bg='#E7ECF7', command=button_run_deepfas)
execute_button.place(x=250, y=300, height = 60, width = 150)
execute_button['font'] = myfont

root.mainloop()