import sys
import os
import pickle
import numpy as np

def write_vtk(fn, v, f):
    len_v = v.shape[0]
    len_f = f.shape[0] 
    
    fp = open(fn, 'w')
    fp.write("# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET POLYDATA\n")
    fp.write(f"POINTS {len_v} float\n")
    for row in v:
        fp.write(f"{row[0]} {row[1]} {row[2]}\n")
    fp.write(f"POLYGONS {len_f} {len_f * 4}\n")
    for row in f:
        fp.write(f"3 {row[0]} {row[1]} {row[2]}\n")
    fp.close()

def main():
    try:
        sys.path.append("meshcnn")
        from mesh import export_spheres
        export_spheres(range(8), "mesh_files")
    except ImportError:
        print("ImportError occurred. Will download precomputed mesh files instead...")
        import subprocess
        dest = "mesh_files"
        if not os.path.exists(dest):
            os.makedirs(dest)
        fname = 'icosphere_{}.pkl'
        for i in range(8):
            url = 'http://island.me.berkeley.edu/ugscnn/mesh_files/' + fname.format(i)
            command = ["wget", "--no-check-certificate", "-P", dest, url]
            try:
                download_state = subprocess.call(command)
            except Exception as e:
                print(e)

        fname_vtk = 'icosphere_{}.vtk'
        for i in range(8):
            try:
                pkl = pickle.load(open(dest + '/' + fname.format(i), "rb"))
                write_vtk(dest + '/' + fname_vtk.format(i), pkl['V'], pkl['F'])
            except Exception as e:
                print(e)

if __name__ == '__main__':
    main()
