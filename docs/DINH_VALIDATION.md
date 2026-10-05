# Dinh: Data, Ground Truth, Projection Validation

## Ket qua frame 000008

Chay `python scripts/validate_sample.py` tai thu muc goc. Script dung bon tep
`data/sample/{000008.pkl,image.png,points.bin,calib.txt}` va ghi vao `results/`.
Lan chay hien tai: **PASS**, anh `1242 x 375`, `17,238` diem LiDAR, toan bo
`17,238` diem co depth duong va nam trong khung hinh. Day la dam may diem demo
da loc theo truong nhin camera; ty le `100%` khong dai dien cho mot scan KITTI day du.

Metadata co `10` annotation: `6` Car hop le, `4` vung `bbox_label=-1` can bo qua.
Trong pickle, ten lop goc cua bon vung nay da bi mat. Cac gia tri sentinel cua
chung phu hop voi quy uoc KITTI DontCare, nhung khong nen khoi phuc ten lop goc
chi tu tep nay. `difficulty=-1` cua mot so xe khong co nghia la bo qua khi
lam diagnostic; no la truong do kho rieng. Neu danh gia theo KITTI official,
can them quy tac loc difficulty/truncation/occlusion cua benchmark.

## Du lieu ban giao

| Tep | Noi dung |
| --- | --- |
| `src/data_utils.py` | Doc pickle gioi han class, extract GT, phep chieu, IoU, membership, association diagnostic |
| `results/objects_000008.json` | 10 object va toan bo thuoc tinh goc, them `object_id`, `class_name`, `ignore`, `valid_3d` |
| `results/objects_000008.csv` | Bang de doc nhanh, `bbox`/`bbox_3d` luu o JSON trong mot o CSV |
| `results/baseline_bbox.png` | Panel tren: 2D GT, vung bo qua; panel duoi: baseline LiDAR, 2D GT, cuboid 3D project |
| `results/projection_validation.json` | Ket qua tung check, checksum SHA-256 input, sai so, thong ke tung object |
| `results/association_000008.npz` | Mang NumPy theo thu tu diem goc, san sang tinh metric |
| `results/association_schema.json` | Shape/dtype tung mang va quy uoc metric |

## Giai ma bbox va nhan

`bbox = (xmin, ymin, xmax, ymax)` theo pixel anh CAM2, dien tich va IoU theo
toa do lien tuc. Phep gan diem 2D dung bien trai/tren bao gom, phai/duoi khong
bao gom. `bbox_3d = (x, y, z, length, height, width, rotation_y)` trong camera
da rectify, don vi met/radian, goc toa do la **tam day hop**. Camera x sang
phai, y xuong duoi, z ve phia truoc. Nhan theo OpenMMLab KITTI v2 converter:
`0 Pedestrian`, `1 Cyclist`, `2 Car`, `3 Van`, `4 Truck`,
`5 Person_sitting`, `6 Tram`, `7 Misc`, `-1 ignore`.

`center_2d` trong pickle la phep chieu cua tam hinh hoc 3D
`(x, y-height/2, z)`, khong phai tam cua rectangle 2D. `depth` la mau so
homogeneous sau P2, khong nhat thiet bang toa do z camera. Ca 6 object
deu khop tam/depth voi tinh lai trong sai so doc lap nho hon `0.001 px`.

## Kiem tra projection

Chuoi chuan: diem Velodyne `xyz1` -> `Tr_velo_to_cam` -> `R0_rect` -> `P2` ->
chia cho toa do homogeneous thu ba. `CAM2.lidar2cam` trong pickle **da co
rectification**; khong nhan `R0_rect` them lan nua khi dung no. Chung toi doi
chieu ma tran `P2 @ R0_rect @ Tr_velo_to_cam` voi `CAM2.lidar2img` trong pickle,
doi chieu tung pixel voi tinh doc lap tu ma tran nguon, va doi chieu ket qua
`src/projection.py`. Sai so pixel lon nhat so voi metadata nguon:
`0.00001703 px`. Sai so pixel lon nhat so voi baseline hien co:
`6.82e-13 px`. Ma tran sai do bo qua rectification lech toi da `13.61 px`;
rectification hai lan lech toi da `13.43 px`. Ket qua ma tran/pixel la kiem tra
tinh nhat quan noi bo, khong phai phep do accuracy ngoai thuc dia.

Cuboid 3D projected bbox co IoU voi 2D GT lan luot: `0.988`, `0.980`,
`0.978`, `0.974`, `0.965`, `0.971`. Khong yeu cau bang 1 vi 2D GT co the
duoc gan nhan theo vat the nhin thay/che khuat va hai bbox bi cat tai bien anh.

## Association metric

Doc bang `np.load("results/association_000008.npz", allow_pickle=False)`.
Hang la chi so zero based cua `points.bin`. Cot cua `membership_3d`,
`membership_kitti_lidar`, `candidate_2d` tuong ung `object_ids=[0,1,2,3,4,5]`.
`membership_3d[i,j]` la diem i nam trong cuboid GT j theo hinh hoc camera;
`candidate_2d[i,j]` la pixel projection cua diem i nam trong rectangle 2D GT j.
`metric_eligible` loai diem ngoai anh, nam trong vung ignore, hoac nam trong
nhieu cuboid 3D. Diem nen con lai duoc giu de tinh false positive. `34` diem
nam trong ignore, `0` diem nam trong nhieu cuboid, `17,204` diem eligible.
Co `1,125` diem thuoc hon mot candidate 2D; khong duoc am tham ep no ve mot
object duy nhat. `reference_object_id=-1` la background, `-2` la ambiguous.

Voi moi object: `TP = count(candidate_2d & membership_3d & eligible)`,
`FP = count(candidate_2d & ~membership_3d & eligible)`,
`FN = count(~candidate_2d & membership_3d & eligible)`. Precision va recall
trong JSON duoc tinh tu ba gia tri nay; khi mau so bang 0, ket qua la `null`.
Day la diagnostic de xem 2D rectangle thu hut nhung diem nao so voi cuboid 3D,
khong phai diem detector, KITTI AP, hay ground truth semantic tung diem.
Khi thu drift, nen giu `membership_3d`/bbox GT co dinh, tinh lai chi `pixels`
va `candidate_2d` voi calibration duoc thu, roi so sanh voi baseline.

Truong `num_lidar_pts` goc la so diem OpenMMLab tinh bang cuboid thang dung
trong he LiDAR. Phep bien doi camera->LiDAR cho tam hop la day du, nhung yaw
duoc doi bang `-rotation_y-pi/2`, bo qua roll/pitch nho cua he camera khi dat
huong hop. Boi vay dem lai voi `membership_3d` trong camera cho ra
`1424, 1940, 878, 668, 53, 164` va khac gia tri pickle. Mang
`membership_kitti_lidar` tai lap dung quy uoc converter va cho ra chinh xac
`1325, 1900, 881, 659, 55, 162`; ca 6 check source count deu PASS.
Khong dung `num_lidar_pts` lam mau so khi metric dung `membership_3d`.

## Nguon va han che

- [MMDetection3D KITTI info converter](https://github.com/open-mmlab/mmdetection3d/blob/main/tools/dataset_converters/update_infos_to_v2.py): mapping nhan, bbox3D, center/depth.
- [MMDetection3D KITTI converter](https://github.com/open-mmlab/mmdetection3d/blob/main/tools/dataset_converters/kitti_converter.py): `num_points_in_gt`.
- [MMDetection3D box operations](https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/structures/ops/box_np_ops.py): `box_camera_to_lidar`, `points_in_rbbox`.
- [MMDetection3D camera box definition](https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/structures/bbox_3d/cam_box3d.py): tam day, truc, yaw.
- [KITTI Vision Benchmark](https://www.cvlibs.net/datasets/kitti/): nguon du lieu va dieu khoan CC BY-NC-SA 3.0.

Day la mot frame demo. Khong co nhan semantic tung diem hoac danh sach detector
prediction, nen chua the tinh metric association cua mot model that hoac
khai quat hoa sang toan bo KITTI. Nguon `.pkl` chi duoc nap tu tep trusted;
pickle tuy y van co nguy co, nen unpickler gioi han constructors va validator
kiem tra frame ID. Kiem thu bang `python -m unittest discover -s tests -v`.
