* images paths をA_tv.hdf5にまとめる
* 普通に学習したモデル(.ckpt.bkup)に対して簡単な文章を入れたらどうなるのかやってみる


### 以下はリマインド
* 学習は1画像に対して5文入れている気がするので、1画像1用のデータセットを試す必要がある

* Data/flowers/jpg とかをいい感じに用意してから、python data_loader.py --data_set="flowers"
  をやる必要がある。
* flower_tv.hdf5を↑でうまく作らないとtrain.pyがまったく機能しない状態で動作してしまうので無意味
* 両方共かなりの時間を必要としているので、注意

### image_06375.txt の中身
* 自然言語文10行の記述
* 

### Data/ImageNet/jpg/*.jpg の概要
* 300*400

