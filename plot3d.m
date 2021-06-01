clear;
close all;
clc;

preds_num = 5000;
iters = 200;

preds_center = readmatrix("preds_center.txt", 'OutputType', 'double');
cx = preds_center(1:preds_num, 1);
cy = preds_center(1:preds_num, 2);
tri = delaunay(cx, cy);

modes = ["navie_iou", "IoU", "GIoU", "DIoU", "CIoU", "LIoU"];

for i = 1:size(modes, 2)
    filename = sprintf("output/loss_%s.txt", modes(i));
    disp(filename);
    loss = readmatrix(filename, 'OutputType', 'double');
    final_loss = loss(:, end);

    figure('Name', modes(i), 'NumberTitle', 'off');
    trimesh(tri, cx, cy, final_loss);
    xlabel('x'); ylabel('y'), zlabel('final error');
    % get rid of white spaces
    set(gca, 'LooseInset', get(gca, 'TightInset'));
    set(gca, 'FontSize', 20);
end