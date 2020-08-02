# Football-Match-Prediction

https://www.kaggle.com/hugomathien/soccer is where our data comes from -> database.sqlite

'1. Data Extraction.ipynb' converts this to csvs
'data_format_pl.ipynb' filters out only those entries relevant to the PL

The benchmark from B365 odds is log loss = 0.620

Currently have 0.6140 (player_attr_to_out_2)




NN:
Train B365 Binary log_loss: 0.615911339001799
Val B365 Binary log_loss: 0.6052028022570637

Models:['model7', 'model16']
Train Accuracy: 0.6715693824118282
Val Accuracy: 0.6610169491525424
Train log loss: 0.6028985969370911
Val log loss: 0.6034085631522174