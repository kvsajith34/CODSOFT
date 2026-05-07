# Data Guide

## Included Dataset
`sample_data.csv` contains 50 labeled movie plot summaries across 8 genres:
- Action, Adventure, Comedy, Fantasy, Horror, Romance, Science Fiction, Thriller

## Format
The CSV must have two columns:
- `plot` — the movie plot summary (string)
- `genre` — the genre label (string)

## Recommended Public Datasets

| Dataset | Source | Size |
|---------|--------|------|
| CMU Movie Summary Corpus | http://www.cs.cmu.edu/~ark/personas/ | ~42,000 movies |
| TMDB Movie Dataset | https://www.kaggle.com/tmdb/tmdb-movie-metadata | ~5,000 movies |
| Wikipedia Movie Plots | https://www.kaggle.com/jrobischon/wikipedia-movie-plots | ~34,000 movies |

## Loading a Custom Dataset
Your CSV must have at minimum:
```
plot,genre
"A young wizard...", Fantasy
"Two detectives...", Thriller
```

Then run:
```bash
python src/train.py --data path/to/your_data.csv
```
