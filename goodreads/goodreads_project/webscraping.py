import pandas as pd
import requests
from bs4 import BeautifulSoup
import re

# initialize dictionary to store dfs for each genre
soup_dict = {'fiction': '', 'mystery-thriller': '', 'historical-fiction': '',
          'fantasy': '', 'romance': '', 'science-fiction': '', 'horror': '',
          'nonfiction': '', 'memoir-autobiography': '', 'history-biography': '',
             'young-adult-fiction': ''}

# cycle through keys
for key in soup_dict:
    # get soup object and add to dictionary
    soup = BeautifulSoup(requests.get(f'https://www.goodreads.com/choiceawards/best-{key}-books-2021').text, features="html.parser")
    soup_dict[key] = soup


def genre_info(genre):
    '''gets necessary info of books of a specified genre from Goodreads 2021 top books website

    Args:
        genre (str): genre of interest

    Returns:
        g_dict (dict): dictionary of all scraped info per genre
    '''
    book_html = soup_dict[genre].find_all(class_='pollAnswer__bookLink')
    link_ls = []
    # add all hrefs to a list
    for book in book_html:
        href_ = book.get('href')
        link = f'https://www.goodreads.com{href_}'
        link_ls.append(link)

    title_ls = []
    author_ls = []
    rating_ls = []
    genres_ls = []
    pgs_ls = []
    descr_ls = []
    g_dict = {'title': title_ls, 'author': author_ls, 'rating': rating_ls, 'genres': genres_ls,
              'pages': pgs_ls, 'description': descr_ls, 'main genre': genre}

    for i in link_ls:
        # get soup obj for each book
        soup_b = BeautifulSoup(requests.get(i).text, features="html.parser")

        title = soup_b.find(class_='Text Text__title1').get('aria-label')
        title = title.replace('Book title: ', '')
        author = soup_b.find(class_='ContributorLink__name').text
        rating = soup_b.find(class_='RatingStatistics__rating').text

        # get list of genres
        genres_temp = soup_b.find('ul', class_="CollapsableList").text
        # separate text by capital letters, cut out first genre which is already included in main genre
        genres = re.findall('[A-Z][a-z]*', genres_temp)[1:]

        # get page number as first element in list
        pgs = soup_b.find('p', {'data-testid': 'pagesFormat'}).text.split(',')[0]

        descr = soup_b.find(class_='BookPageMetadataSection__description').text
        # remove unwanted string in beginning of some descriptions
        unwanted_str = 'An alternative cover edition for this ISBN can be found here.'
        if unwanted_str in descr:
            descr = descr[61:]

        # add info to lists
        title_ls.append(title)
        author_ls.append(author)
        rating_ls.append(rating)
        genres_ls.append(genres)
        pgs_ls.append(pgs)
        descr_ls.append(descr)

    # fix bug with LGBT being separated
    for ls in genres_ls:
        if 'L' in ls:
            ls.remove('L')
            ls.remove('G')
            ls.remove('B')
            ls.remove('T')
            ls.append('LGBT')
    return g_dict


fiction_dict = genre_info('fiction')
fiction_df = pd.DataFrame(fiction_dict)

mys_thr_dict = genre_info('mystery-thriller')
mys_thr_df = pd.DataFrame(mys_thr_dict)

hist_fic_dict = genre_info('historical-fiction')
hist_fic_df = pd.DataFrame(hist_fic_dict)

fantasy_dict = genre_info('fantasy')
fantasy_df = pd.DataFrame(fantasy_dict)

romance_dict = genre_info('romance')
romance_df = pd.DataFrame(romance_dict)

scifi_dict = genre_info('science-fiction')
scifi_df = pd.DataFrame(scifi_dict)

horror_dict = genre_info('horror')
horror_df = pd.DataFrame(horror_dict)

nonfic_dict = genre_info('nonfiction')
nonfic_df = pd.DataFrame(nonfic_dict)

memoir_dict = genre_info('memoir-autobiography')
memoir_df = pd.DataFrame(memoir_dict)

histbio_dict = genre_info('history-biography')
histbio_df = pd.DataFrame(histbio_dict)

yafic_dict = genre_info('young-adult-fiction')
yafic_df = pd.DataFrame(yafic_dict)

df_ls = [fiction_df, mys_thr_df, hist_fic_df, fantasy_df, romance_df, scifi_df, horror_df, nonfic_df, memoir_df,
         histbio_df, yafic_df]
full_df = pd.DataFrame()

for df in df_ls:
    full_df = pd.concat([full_df, df])

num_pages = []
for val in full_df["pages"]:
    # change page numbers to int
    intpgs = int(val.split(' ')[0])
    num_pages.append(intpgs)

full_df['pages'] = num_pages
full_df.to_csv("goodreads.csv", index = False)
print(full_df)