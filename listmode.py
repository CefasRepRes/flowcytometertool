#!/usr/bin/env python3

# A partial implementation of "list mode" for a given JSON CYZ file.
# Outputs a CSV file with summary data for each of the particles.

import json
import statistics
import pandas as pd
import sys
import os
import math
import base64
from io import BytesIO
from PIL import Image
import os.path

def extract(particles, dateandtime, images = '', save_images_to = ''):
    lines = []
    images_passed = False
    all_particle_ids = []
    if images != '':
        images_passed = True
        # Assuming data is your dictionary containing images
        all_particle_ids = [image['particleId'] for image in images]
        indices_of_images = [index for index, image in enumerate(images)]
    for particle in particles:
        line = {}
        line["id"] = particle["particleId"]
        if particle["particleId"] in all_particle_ids:
            index_of_match = all_particle_ids.index(particle["particleId"])
            image_data = base64.b64decode(images[index_of_match]['base64'])
            image = Image.open(BytesIO(image_data))
            try:
                image.save(save_images_to+str(particle["particleId"])+'.tif')
            except:
                print('Failed to save. Didz you pass images without specifying an image directory?')
            #plt.imshow(imagei, cmap='gray')
            #plt.axis('off')  # Turn off axis
            #plt.show()
        line["datetime"] = dateandtime
        for parameter in particle["parameters"]:
            description = parameter["description"]
            line[f"{description}_length"] = parameter["length"]
            line[f"{description}_total"] = parameter["total"]
            line[f"{description}_maximum"] = parameter["maximum"]
            line[f"{description}_average"] = parameter["average"]
            line[f"{description}_inertia"] = parameter["inertia"]
            line[f"{description}_centreOfGravity"] = parameter["centreOfGravity"]
            line[f"{description}_fillFactor"] = parameter["fillFactor"]
            line[f"{description}_asymmetry"] = parameter["asymmetry"]
            line[f"{description}_numberOfCells"] = parameter["numberOfCells"]
            line[f"{description}_sampleLength"] = parameter["sampleLength"]
            line[f"{description}_timeOfArrival"] = parameter["timeOfArrival"]
            line[f"{description}_first"] = parameter["first"]
            line[f"{description}_last"] = parameter["last"]
            line[f"{description}_minimum"] = parameter["minimum"]
            line[f"{description}_swscov"] = parameter["swscov"]
            line[f"{description}_variableLength"] = parameter["variableLength"]
        lines.append(line)
    return lines


# filename = "C:/Users/JR13/Documents/LOCAL_NOT_ONEDRIVE/lucinda-flow-cytometry/data/interim/json/nano_cend16_20 2020-10-09 00u03.cyz.json"

# filename = "C:/Users/JR13/Documents/LOCAL_NOT_ONEDRIVE/lucinda-flow-cytometry/temp/tempfile.json"

def main(filename,fileout,save_images_to = ''):
    #print(os.path.dirname(filename))
    data = json.load(open(filename, encoding="utf-8-sig"))
    lines = extract(particles=data["particles"],dateandtime = data["instrument"]["measurementResults"]["start"],images = data["images"], save_images_to = save_images_to+'/'+data["filename"])
    df = pd.DataFrame(lines)
    print("save to " +fileout)
    df.insert(loc=0, column="filename", value=os.path.basename(filename))
    df.to_csv(fileout, index=False)
    #outfile = os.path.basename(fileout)
    #print("saving to " + outfile)
    #df.to_csv(outfile, index=False)
    #df.to_csv(f"{outfile}.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        if len(sys.argv) != 6:
            print("2 or 3 command line arguments expected in the form ['../dir/listmode.py', '../in/nano_cend16_20 2020-10-09 00u03.cyz.json', '--output', '../out/nano_cend16_20 2020-10-09 00u03.cyz.json.csv', '--imagedir','../imagesout/']")
            sys.exit(1)
    main(sys.argv[1],sys.argv[3],sys.argv[5])