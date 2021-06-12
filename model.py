from utils import *
import pandas as pd
from wordcloud import WordCloud
#Pegamos las dos listas
listTitle_Abstract = {'titles': getalltitles(), 'abstract': getalldocs()}
#Convertimos la lista en un data frame
abstTitleDF = pd.DataFrame(listTitle_Abstract, columns=['titles', 'abstract'])
# Pegamos todos las palabras procesadas.
long_string = ','.join(list(abstTitleDF['titles'].values))
long_string += ','.join(list(abstTitleDF['abstract'].values))
wordcloud = WordCloud(background_color="white", max_words=10000, contour_width=3, contour_color='steelblue')
# Generamos un word cloud
wordcloud.generate(long_string)
# Visualizaci'on del word cloud
image = wordcloud.to_image()
image.show()
image.save('cloud.png')