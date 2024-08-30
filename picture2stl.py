import argparse
import cv2
import numpy as np
from stl import mesh
from tqdm import tqdm


def create_stl(img_path, stl_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    objs = []
    for x in tqdm(range(img.shape[0])):
        for y in range(img.shape[1]):
            vertices = np.array([
                [x, y, 0],
                [x + 1, y, 0],
                [x + 1, y + 1, 0],
                [x, y + 1, 0],
                [x, y, 16 - int(img[x, y] / 16)],
                [x + 1, y, 16 - int(img[x, y] / 16)],
                [x + 1, y + 1, 16 - int(img[x, y] / 16)],
                [x, y + 1, 16 - int(img[x, y] / 16)],
            ])
            faces = np.array([
                [0, 3, 1],
                [1, 3, 2],
                [0, 4, 7],
                [0, 7, 3],
                [4, 5, 6],
                [4, 6, 7],
                [5, 1, 2],
                [5, 2, 6],
                [2, 3, 6],
                [3, 7, 6],
                [0, 1, 5],
                [0, 5, 4],
            ])
            obj = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
            for i, f in enumerate(faces):
                for j in range(3):
                    obj.vectors[i][j] = vertices[f[j], :]
            objs.append(obj.data.copy())
    merged = mesh.Mesh(np.concatenate(objs))
    merged.save(stl_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('imgpath')
    parser.add_argument('stlpath')
    args = parser.parse_args()
    create_stl(args.imgpath, args.stlpath)


if __name__ == "__main__":
    main()
