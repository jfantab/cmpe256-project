# cmpe256-project

Group members: John Lu, Hardy Leung, Carlos Hernandez

Our final project was on the [Million Songs dataset](http://millionsongdataset.com/). Listed below are the links to the respective datasets we used. To reproduce our
results, please review and follow the instructions to download and unzip necessary files.

Link to the [Kaggle challenge](https://www.kaggle.com/competitions/msdchallenge/data).

1. [Taste Profile Subset](http://millionsongdataset.com/tasteprofile/) | [Train data](http://millionsongdataset.com/sites/default/files/challenge/train_triplets.txt.zip) 
- Note: These train and test files exceed Github's max file size limit, so the links are provided here for download.  To reproduce our results, please download and unzip the file in the million_songs_data directory.
2. [MusixMatch Data](http://millionsongdataset.com/musixmatch/) | [Train split](http://millionsongdataset.com/sites/default/files/AdditionalFiles/mxm_dataset_train.txt.zip) | [Test split](http://millionsongdataset.com/sites/default/files/AdditionalFiles/mxm_dataset_test.txt.zip)
- Note: The train and test split is arbitrary, both files can be combined to get the full coverage of lyrics for all the songs. These zip files are already included in the repository.
3. [TagTraum Genre Annotations](http://www.tagtraum.com/msd_genre_datasets.html) | [Data link](https://www.tagtraum.com/genres/msd_tagtraum_cd1.cls.zip)
- Note: These zip files are already included in the repository.

4. [Kaggle Challenge Dataset I](https://www.kaggle.com/competitions/msdchallenge/data)
- Note: These zip files are already included in the repository. To reproduce our results, please cd to the million_songs_data directory, and:
- unzip msdchallenge.zip
- unzip taste_profile_song_to_tracks.txt.zip

5. [Kaggle Challenge Dataset II](http://millionsongdataset.com/challenge/#data1) | [Data link](http://millionsongdataset.com/sites/default/files/challenge/EvalDataYear1MSDWebsite.zip)
- Note: These zip files are already included in the repository. To reproduce our results, please cd to the million_songs_data directory, and:
- unzip test_visible.txt.zip
- unzip test_hidden.txt.zip
- Note: We also need the actual utility matrices for the testset, both the visible part and the hidden part, which are test_visible.txt and test_hidden.txt respectively. The file test_visible.txt is actually identical to kaggle_visible_evaluation_triplets.txt in the Kaggle Challenge dataset, as the visible half of the song preference is needed for the competition. However, the hidden half is hidden from the competitors, and is used for our evaluation of the testset only

The required libraries used in the preprocessing can be downloaded with the `pip install -r requirements.txt` command. The preprocessing notebooks are made so that the file structure and necessary files can be downloaded and reconstructed.

To run the MSD code, make sure you have following the above instructions to download and unzip all the necessary files. For Lyrics embeddings flow, please
run John's embedding generation code to generate a file called song_embeddings.csv and copy the file to the data directory first.

To run the algorithms, first cd to the cpp directory, and first create the executable.

cd /path/to/cpp
make msd

You can run "item", "user", "popularity", "embedding", and "ensemble". For example:

make item

If you want to run ALS or LMF, you need to have run an extra step which is to run the Python Implicit package to generate the embeddings first. This needs to be done once.

make implicit

Afterwards you can run "make als" or "maks lmf".
