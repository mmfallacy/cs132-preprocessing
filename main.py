import polars as pl
from nltk.corpus import stopwords

# Do not forget to call nltk.download("stopwords") on install

tlsw = []
ensw = stopwords.words('english')
with open("./stopwords-tl.txt", "r") as raw_stopwords:
    tlsw.extend(raw_stopwords.read().split("\n"))

print(ensw)

df = pl.read_csv("dataset.csv")

# Clean ID Part
df = df.select([
    pl.col("ID").apply(lambda id: id.split("-")[1]).cast(pl.Int32),
    pl.exclude("ID")
])

# Remove rows without tweet AND tweet URL
df = df.filter(pl.col("Tweet").is_not_null() & pl.col("Tweet URL").is_not_null())

# Include only necessary columns
df = df.select([
    pl.col("ID"),
    pl.col("Tweet")
])

# Change tweet case to lowercase
df = df.select([
    pl.exclude("Tweet"),
    pl.col("Tweet").apply(lambda tweet: tweet.lower())
])

# Cast Tweets to word array instead of long string.
df = df.select([
    pl.exclude("Tweet"),
    pl.col('Tweet').apply(lambda tweet: tweet.split()).cast(pl.List(str))
])

# Strip tagalog and english stopwords
df = df.select([
    pl.exclude("Tweet"),
    pl.col("Tweet").arr.eval(pl.element().filter(~pl.element().is_in(tlsw) & ~pl.element().is_in(ensw)), parallel=True)
])



print(df)
print(df.row(by_predicate=(pl.col("ID")==3)))

# df = df.select([
#     pl.exclude("Tweet"),
#     pl.col("Tweet").apply(lambda words: " ".join(words)).cast(str)
# ])
# df.write_csv("clean.csv")
#
