#%%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import cv2

def load_images(path: str, file_ending: str=".png") -> (list, int, int):
    images = []

    files = os.listdir(path)
    files.sort()

    for i in files:
        if not i.endswith(file_ending):
            continue
        image = np.asarray(np.float64(mpl.image.imread(path+"/"+i)))
        images.append(image)

    dimension_y, dimension_x = images[0].shape

    return images, dimension_x, dimension_y


def setup_data_matrix(images: list) -> np.ndarray:
    n = len(images)
    m = images[0].shape[0] * images[0].shape[1]

    D = np.zeros((n, m))

    for i, image in enumerate(images):
        D[i] = image.flatten()


    return D


def calculate_pca(D: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    mean_data = np.mean(D, axis = 0)

    _, svals, pcs = np.linalg.svd(D-mean_data, full_matrices=False)

    return pcs, svals, mean_data


def accumulated_energy(singular_values: np.ndarray, threshold: float = 0.8) -> int:
    k = 0
    sum = np.sum(singular_values)

    temp = 0

    while threshold > temp/sum:
        temp += singular_values[k]
        k+=1


    return k


def project_faces(pcs: np.ndarray, images: list, mean_data: np.ndarray) -> np.ndarray:
    n = len(images)
    m = pcs.shape[0]
    coefficients = np.zeros((n,m))

    images = setup_data_matrix(images) - mean_data

    for i in range(n):
        coefficients[i] = np.dot(pcs, images[i])

    return coefficients


def identify_faces(coeffs_train: np.ndarray, pcs: np.ndarray, mean_data: np.ndarray, path_test: str) -> (np.ndarray, list, np.ndarray):
    imgs_test, _, _ = load_images(path_test)

    coeffs_test = project_faces(pcs, imgs_test, mean_data)

    n = coeffs_train.shape[0]
    m = len(imgs_test)
    scores = np.zeros((n, m))
    
    for i in range(n):
        for j in range(m):
            scores[i,j] = np.arccos(np.dot(coeffs_train[i], coeffs_test[j]) / 
                (np.linalg.norm(coeffs_train[i]) * np.linalg.norm(coeffs_test[j])))
            

    return scores, imgs_test, coeffs_test

images, dimx, dimy = load_images("./data/train")
data_matrix = setup_data_matrix(images)
pcs, svals, mean_data = calculate_pca(data_matrix)

k = accumulated_energy(svals)
pcs = pcs[:k]

#traincoefs = project_faces(pcs, images, mean_data)

#scores, testimages, testcoeffs = identify_faces(traincoefs, pcs, mean_data, "./data/test")


cap = cv2.VideoCapture(0)
while(True):
  ret, frame = cap.read()
  frame = cv2.resize(frame, (dimx,dimy))
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  cv2.imshow('frame',frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()

# %%
