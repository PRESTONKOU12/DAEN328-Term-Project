# Movies in the chicago parks 
Authors:
- Diya Dev
- Preston Kouyoumdjian
- Andres Sanchez
- Will Youngblood 


## Overview:
The goal of this project is to develop insights into different movie events within various parks in Chicago.  With the end goal to derive data driven insights to answer the following questions: Grouping by zip-code, what genres are the most popular, Can we compare events hosted to income / zip-code, and many more questions.

The architecture of the pipeline follows a standard ETL (extract, transform, load) system.  Ingesting data from the chicago data portals' "Movies in the park: (year)" for years 2014 to 2019.  To develop a more nuanced analysis, we used multiple datasets across multiple years. 


## Pipeline structure
The steps of our pipeline are as follows
- Extract: 
    - Use rest api to extract 6 years of movie data.
    - Store into individual tables
    - Combine into 1 large database (no feature cleaning)
- Transform: 
    - Clean individual features.
- Load:
    - Seperate clean data into db schema (to ensure 3NF) 

## How to run
```bash
    docker-compose up --build
```
