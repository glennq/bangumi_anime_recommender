Bangumi Anime Recommender
=========================
This is a project out of my personal interest. I am going to build a
recommender system for anime on bangumi.tv, for which the major
motivation is for personal use.

For the first step, I am going to do web crawling and get user rating
data for all anime entries. For the purpose of my project, I will only
be interested in users who have rated a number of anime. And only
ratings in "watched" and "dropped" are considered since such ratings
should be finalized.

For implementation of the recommender system, I am planning to apply
the following three algorithms:

1. Content-based system, which makes use of the tags for each anime

2. User similarity based collaborative filtering, for which I might
try cosine similarity

3. Matrix factorization based collaborative filtering

I may also try to add temporal effects in later versions
