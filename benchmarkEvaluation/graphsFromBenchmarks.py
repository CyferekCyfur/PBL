import matplotlib as mp


def extractData(file):
    dataBbox = {}  # dict for bounding box efficiency data
    dataBev = {}  # dict for bounding box efficiency data
    data3d = {}  # dict for 3D object detection efficiency data for
    dataAos = {}  # dict for orientation efficiency data
    for line in file:
        if line.startswith("Car"):
            difficultyLevels = []
            beforeAndAfterAt = line.split("@")
            noSuffix = beforeAndAfterAt[1].removesuffix(":\n")
            difficultyLevels = [
                float(element.strip()) for element in noSuffix.split(",")
            ]

            #    for line in file:
    #        if line.startswith('bbox'):
    #    for line in file:
    #        if line.startswith('bev'):
    #    for line in file:
    #        if line.startswith('3d'):
    #    for line in file:
    #        if line.startswith('aos'):

    return dataBev, dataBbox, data3d, dataAos


if __name__ == "__main__":
    smoke = open("SMOKE", "r")
    benchmarkMonocon12_17 = open("benchmarkMonocon12_17", "r")
    yolo = open("YOLOstereo3D", "r")

    extractData(yolo)
    extractData(smoke)
