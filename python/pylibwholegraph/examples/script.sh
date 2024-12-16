python benchmark.py --feat_dim 128 --batch_size 64 --node_count 5000000
python benchmark.py --feat_dim 256 --batch_size 64 --node_count 5000000
python benchmark.py --feat_dim 512 --batch_size 64 --node_count 5000000
python benchmark.py --feat_dim 1024 --batch_size 64 --node_count 5000000


python benchmark.py --feat_dim 128 --batch_size 128 --node_count 5000000
python benchmark.py --feat_dim 128 --batch_size 256 --node_count 5000000
python benchmark.py --feat_dim 128 --batch_size 512 --node_count 5000000

python benchmark.py --feat_dim 128 --batch_size 64 --node_count 10000000
python benchmark.py --feat_dim 128 --batch_size 64 --node_count 20000000
python benchmark.py --feat_dim 128 --batch_size 64 --node_count 40000000