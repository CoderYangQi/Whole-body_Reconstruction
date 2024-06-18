path = "C:\\Users\\chaoyu\\Documents\\projects\\VISoR-data-analysis\\VISoR_Reconstruction\\tools\\imagej_bridge\\run.bat"
f = getInfo("image.filename");
d = getInfo("image.directory");
f = split(f, ".");
in = d + f[0] + ".align.tar";
out = getDirectory("temp") + f[0] + ".roi.mha"
Dialog.create("Reconstruct ROI");
Dialog.addString("Imagej bridge path:", path);
Dialog.addNumber("Result voxel size:", 1.0);
Dialog.addString("Output file:", out);
Dialog.show();
getSelectionBounds(x, y, w, h);
path = Dialog.getString();
voxel_size = Dialog.getNumber();
out = Dialog.getString();
print(path + " " + in + " " + out + " " + x + " " + y + " " + w + " " + h + " " + voxel_size)
v = exec(path,in,out,x,y,w,h,voxel_size);
open(out);
