import os
import numpy as np
import model_analysis



run = {}
machine = {}
model = {}

path = os.getcwd()
for thing in os.listdir(path + "/Saved"):
    model["name"] = thing
    for run_settings in os.listdir(path + "/Saved/" + thing):
        entries = run_settings.split("[")
        search = entries[1].split("]")[0]
        if entries[2][0] == "4":
            start = 4
            end = 25
        else:

            if entries[2][0] == "-":
                start = eval(entries[2][:4])
                end = eval(entries[2][5:-1])
            else:
                start = eval(entries[2][:3])
                end = eval(entries[2][4:-1])
        
        run["epochs"] = eval(entries[3][2:-1])
        machine["visual_n"] = entries[4][2:-1]
        model["args"] = {}
        if entries[2][0] == "4":
            sstart = 4
        else:
            sstart = 5
        for entry in entries[sstart:]:
            var = ""
            val = ""
            flag = 0
            for char in entry:
                if char == "=":
                    flag = 1
                if not flag:
                    var += char
                else:
                    val += char
            val = eval(val[1:-1])
            model["args"][var] = val

        run_path =f"{path}/Saved/{thing}/{run_settings}"
        error = np.load(run_path + "/error.npy")
        length = len(error)
        if entries[2][0] == "4":
            sarray = np.linspace(start, end, length, dtype=int)
        else: 
            sarray = np.linspace(start, end, length)
    
        model_analysis.plot(
            run,
            machine,
            model,
            search,
            sarray,
            show=False
        )

