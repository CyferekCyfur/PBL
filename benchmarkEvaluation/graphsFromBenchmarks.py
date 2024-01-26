import re
import matplotlib.pyplot as plt

def extractData(file):
    dataBbox = []  # dict for bounding box efficiency data
    dataBev = []  # dict for bounding box efficiency data
    data3d = []  # dict for 3D object detection efficiency data for
    dataAos = []
    difficulties = []
    # Define a regular expression pattern to extract Car data
    pattern = re.compile(
        r"Car AP40@(\d+\.\d+), (\d+\.\d+), (\d+\.\d+):\n"
        r"bbox AP40:(\d+\.\d+), (\d+\.\d+), (\d+\.\d+)\n"
        r"bev  AP40:(\d+\.\d+), (\d+\.\d+), (\d+\.\d+)\n"
        r"3d   AP40:(\d+\.\d+), (\d+\.\d+), (\d+\.\d+)\n"
        r"aos  AP40:(\d+\.\d+), (\d+\.\d+), (\d+\.\d+)"
    )

    # Find all matches in the data
    matches = pattern.findall(file)
    # print(matches)
    # Display the results
    
    for match in matches:
        # print(match[6:9])
        # print("Car AP40@{}:".format(match[0]))
        # print("  Bbox AP40:", ", ".join(match[3:6]))
        dataBbox.append(match[3:6])
        #    print("  Bev  AP40:", ", ".join(match[6:9]))
        dataBev.append(match[6:9])
        #    print("  3d   AP40:", ", ".join(match[9:12]))
        data3d.append(match[9:12])

        #    print("  Aos  AP40:", ", ".join(match[12:]))

        dataAos.append(match[12:])
        difficulties = match[0]
        print(difficulties)
        
        print(dataBbox)
        print(dataBev)
        print(data3d)
        print(dataAos)


if __name__ == "__main__":
    #    smoke = open("SMOKE", "r")
    #    benchmarkMonocon12_17 = open("benchmarkMonocon12_17", "r")
    #yolo = open("YOLOstereo3D", "r")
    smoke = """
Pedestrian AP40@0.50, 0.50, 0.50:
bbox AP40:57.8818, 48.5635, 40.6865
bev  AP40:5.1800, 4.0200, 3.2564
3d   AP40:3.7722, 2.9445, 2.5435
aos  AP40:31.21, 26.30, 22.04
Pedestrian AP40@0.50, 0.25, 0.25:
bbox AP40:57.8818, 48.5635, 40.6865
bev  AP40:25.8456, 21.0090, 17.6305
3d   AP40:25.2026, 20.3302, 16.2586
aos  AP40:31.21, 26.30, 22.04
Cyclist AP40@0.50, 0.50, 0.50:
bbox AP40:46.8245, 28.0950, 25.1727
bev  AP40:3.0334, 1.4865, 1.4771
3d   AP40:1.9600, 1.2703, 0.9835
aos  AP40:27.86, 16.04, 14.35
Cyclist AP40@0.50, 0.25, 0.25:
bbox AP40:46.8245, 28.0950, 25.1727
bev  AP40:16.4194, 9.0275, 7.9642
3d   AP40:16.1095, 8.8931, 7.8313
aos  AP40:27.86, 16.04, 14.35
Car AP40@0.70, 0.70, 0.70:
bbox AP40:94.4864, 86.2933, 79.2313
bev  AP40:24.5131, 18.5286, 16.3375
3d   AP40:16.7713, 12.8792, 10.7311
aos  AP40:91.59, 82.51, 74.49
Car AP40@0.70, 0.50, 0.50:
bbox AP40:94.4864, 86.2933, 79.2313
bev  AP40:58.6409, 43.5922, 39.2054
3d   AP40:53.6863, 39.8296, 34.5969
aos  AP40:91.59, 82.51, 74.49
"""
 
    extractData(smoke)
    plt.subplot(4,1,1)
    plt.plot()
