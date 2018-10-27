import numpy as np
import matplotlib.pyplot as plt
def main():
    data = {}
    tensorShape = (100, 12, 207, 1)
    # tensorShape = (5, 2, 2, 1)
    data["x"] = np.zeros(tensorShape)
    data["y"] = np.zeros(tensorShape)
    xInputs = np.array(range(tensorShape[1] * tensorShape[2])).reshape((tensorShape[1], tensorShape[2]))
    xInputs = np.expand_dims(xInputs, axis=2)
    yInputs = (np.array(range(tensorShape[1] * tensorShape[2])) + (tensorShape[1] * tensorShape[2])).reshape((tensorShape[1], tensorShape[2]))
    yInputs = np.expand_dims(yInputs, axis=2)
    xInputs = xInputs * np.pi / 4
    yInputs = yInputs * np.pi / 4
    for instance in range(1, tensorShape[0] + 1):
        outX = np.sin(xInputs + 2*instance)
        data["x"][instance-1,...] = outX
        outY = np.sin(yInputs + 2*instance)
        data["y"][instance-1,...] = outY
    # for i in range(5):
    #     plt.clf()
    #     plt.scatter(range(tensorShape[1] * tensorShape[2]), data["x"][i].flatten(), color="red")
    #     plt.scatter(np.array(range(tensorShape[1] * tensorShape[2])) + tensorShape[1] * tensorShape[2], data["y"][i].flatten(), color="blue")
    #     plt.show()
    np.savez("./data/trainSine.npz", x = data["x"][:70], y=data["y"][:70])
    np.savez("./data/valSine.npz", x = data["x"][70:90], y=data["y"][70:90])
    np.savez("./data/testSine.npz", x = data["x"][90:], y=data["y"][90])
if __name__ == '__main__':
    main()