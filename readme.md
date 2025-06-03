# Python Tools for CYZ File Processing

There are a few python tools here in various states. This repository resembles an attempt to put them in one place, anticipating that we will be using a random forest model to classify our data.
It is developed around a handful of labelled flow cytometer files held in https://citprodflowcytosa.blob.core.windows.net/public/exampledata/ but you could export your own data from cytoclus software and put them in flowcytometertools/exampledata/. To do this in cytoclus, first select your file, with sets defined, under "Database" click Exports and check the box for CYZ file (for set)
This was developed on windows but a github actions workflow tests whether the `Download & Train` tab will work on a linux machine.
Ideally any users should be familiar with python because in all likelihood something will break. 

Acknowledgements to:
Lucinda Lanoy for her masters work in custom_functions_for_python.py https://github.com/CefasRepRes/lucinda-flow-cytometry on model training.
Sebastien Galvagno, Eric Payne and Rob Blackwell for their parts played in cyz2json (flowcytometertool uses https://github.com/OBAMANEXT/cyz2json/releases/tag/v0.0.5)
OBAMA-NEXT Data labellers Veronique, Lumi, Zeline, Lotty and Clementine.


## Download

Check releases to download a "distributable", unzip this, open software via the distributable\trainer_lucinda_version\trainer_lucinda_version.exe
Whilst tested on linux, I have only made pyinstaller builds of the software for windows.
https://github.com/CefasRepRes/flowcytometertool/releases


## Compilation

This was developed in miniforge3 prompt. You can download this to your machine and compile the program with the command "pyinstaller trainer_lucinda_version.spec"

## Github actions

For reproducibility and to test whether this can to run on another machine we use github actions on an ubuntu runner, in git bash terminal you can trigger a test build on github with a new VERSION, e.g:
VERSION="0.0.0.1"; git tag -a v$VERSION -m "Release version $VERSION"; git push origin v$VERSION


---

## Tabs Overview 

### 1: `Download & Train`
- Takes functions developed under Lucinda Lanoy's Masters research inside a tkinter GUI wrapper instead of the R markdown wrapper Lucinda used. It is a stripped back version.
- Trains models including random forest using `sklearn`.
- At the time of writing, this is the only part tested using GitHub Actions.
- Tested on a Linux runner, but also runs locally on Windows.
- *issue* Ihave not tested building a release for a while

---

### `Visualise & Label`
- After you have trained a model and clicked test classifier, you can move on to the next tab.
- *issue* FIRST SELECT X and Y AXES then click update plot otherwise you get KeyError: ''
- *issue* By default the colour will be the predicted label (*issue* it can crash if you change this to another column that has too many unique values).
- Has a facility to re label the data too
- You can explore axis options
- *issue* This aspect is also broken now: Since it is just a dataframe visualiser, it is possible to load another csv file to explore (like the mixfile - explained later)

---


### `make mixfile`
- *issue* It makes sense to put this into the blob processor as it uses SAS token, we can use this definition for both the mixfile and the blob processor
- *issue* untested lately
- Takes 1-in-1000 shuffle of all the particles in a load of PROCESSED predictions held on the blob store and saves it in a CSV file that the visualiser can plot.
- The idea here is that we could have the training dataset defined in one file that captures all the variability in the water, by taking particles from every file.

---

### `Process blob container`
- Uses the trained Python model.
- Operates on a blob store full of CYZ files.
- Installs `cyz2json` and Python requirements.
- Can download a file from blob store.
- Can extract CYZ to JSON.
- Then gets the listmode parameters from this JSON.
- Then uses R random forest to infer on those listmode parameters.
- Uses Python to make a 3D plot.
- Uploads this lot up to a blob destination.
- *issue* only source directory is an input, both source and destination AND the SAS key location must be entered
- *issue* requires both SAS token and blob access which is not very user friendly
- *issue* throws an error if it has been cancelled half way through and the files from last time still exist
- *issue* process log contains the SAS token, remove this leakage

---

### `Local watcher`
- Uses the trained Python model.
- Watches a directory into which CYZ files are going to appear (be dropped into).
- When one appears, applies the Python model and does all the other processing above including the 3D plot in a destination directory.

---
