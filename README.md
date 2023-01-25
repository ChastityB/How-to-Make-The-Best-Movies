![alt text](https://www.themoviedb.org/assets/2/v4/logos/v2/blue_long_1-8ba2ac31f354005783fab473602c34c3f4fd207150182061e425d366e4f34596.svg)

# How to Make The Best Movies
<a href='https://public.tableau.com/views/TMDBMovies_16732589494920/HomePage?:language=en-US&:display_count=n&:origin=viz_share_link'> TMDB Movie Dashboard Tableau Link</a>

![Screen Shot 2023-01-24 at 8 25 05 PM](https://user-images.githubusercontent.com/97481574/214459421-4e0c8c75-afcc-45f7-870c-ac5842a2ee32.png)

# Business Problem
Analyze IMDB's extensive publicly available dataset, extract the financial datfrom TMDB's API and extract insights and recommendations on how to make the best movies. 

This project uses a combination of machine learning model based insights and hypothesis testing to answer stakeholder questions. A Tableau Dashboard was created to supplement visually the stakeholder questions. 

# Stakeholder's Requirements. 
* Stackholders did not want to include movies released before 2000. 
* Stackholders are only interested in movies released in the United States

# Part 1 - IMDB Data Processing

* Download movie metadata from IMDB's public datasets. 
  - The datasets can be obtained <a href='https://datasets.imdbws.com/'> here </a>. 
  - Data dictionary for extracted datasets can be obtained <a href='https://www.imdb.com/interfaces/'> here </a>. 
* Filter the necessary information that meets the stakeholder's requirements
* Files extracted include: 
  - title.basics.tsv.gz
  - title.ratings.tsv.gz
  - title.akas.tsv.gz
  - title.crew.tsv.gz
  - title.principals.tsv.gz
  - name.basics.tsv.gz

# Part 2 - Extracting TMDB Data
* Extract financial data and MPAA rating for the movies using TMDB's API. 
## Exploratory Data Analysis 
### Movies with Financial Data
![Screen Shot 2023-01-24 at 8 47 44 PM](https://user-images.githubusercontent.com/97481574/214462425-8783e08c-ee5f-46f5-989e-132006dc5107.png)

### Movies in each Category
![Screen Shot 2023-01-24 at 8 47 53 PM](https://user-images.githubusercontent.com/97481574/214462480-201680f7-e21f-43a8-9045-6563fe435256.png)

### Average Revenue per Certification
![Screen Shot 2023-01-24 at 8 48 01 PM](https://user-images.githubusercontent.com/97481574/214462528-f718d55b-934d-4c10-8630-9966a5ea7888.png)

### Average Budget per Certification
![Screen Shot 2023-01-24 at 8 48 10 PM](https://user-images.githubusercontent.com/97481574/214462558-70e2d678-59da-47bb-bea6-cab14f3fb2cf.png)

# Part 3 - MySQL Database

* Normalize all extracted IMDB movie data into proper MySQL database
  - See notebook for more details on this process

# Part 4 - Hypothesis Testing
(to be updated)
