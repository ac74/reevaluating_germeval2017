#!/usr/bin/env python3
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

### function to create pre-processed dataframe used for subtask A and B --> outer_df ###
def create_outer_df(root):
    """ 
    create dataframe to work with out of outer tree (=extract only informations on document level).

    Arg: root: created out of XML file using tree.getroot()
    """

    columns = ["id", "text", "relevance", "sentiment", "opinion"]
    df = pd.DataFrame(columns = columns)
    for node in root:
        id = node.attrib.get("id")
        text = node.find("text").text
        relevance = node.find("relevance").text if node is not None else np.nan
        sentiment = node.find("sentiment").text if node is not None else np.nan
        opinion = node.find("Opinions") is not None # extracts flag for opinion
        df = df.append(pd.Series([id, text, relevance, sentiment, opinion], index = columns), ignore_index = True)
    return df

### functions to create pre-processed dataframe used for subtask C and D --> df ###
def iter_opinions(document):
    """ 
    iterate over opinions in a document. Return dictionary with subnode entries.

    Arg: 
      document: part of tree (subnode) extracted from XML file
    """
    document_id = document.attrib.get("id")
    document_text = document.find("text").text
    document_relevance = document.find("relevance").text
    document_sentiment = document.find("sentiment").text
    document_attr = {"id": document_id, "text": document_text, "relevance": document_relevance, "sentiment": document_sentiment}

    if document.find('Opinions') is None:
        opinion_dict = document_attr.copy()
        opinion_dict.update({"category":np.nan, "from":np.nan, "to":np.nan, "target":np.nan, "polarity":np.nan})
        yield opinion_dict
    else:
        for opinion in document.iter('Opinion'):
            opinion_dict = document_attr.copy()
            opinion_dict.update(opinion.attrib)
            yield opinion_dict

def iter_docs(tree):
    """
    iterate over documents in tree.

    Arg: 
      tree: a tree extracted from .xml file (using ElementTree)
    """
    for document in tree.iter('Document'):
        for row in iter_opinions(document):
            yield row

def convert_df(tree):
    """
    convert tree into dataframe, add aspect variable and correct data entries.

    Arg: 
      tree: a tree extracted from .xml file (using ElementTree)
    """
    df = pd.DataFrame(list(iter_docs(tree)))
    # add aspect variable
    aspect = df.category.str.split('#').str[0]
    df["aspect"] = aspect
    # replace wrongly written entries for polarity
    df.polarity = df.polarity.replace('positve', 'positive') # comes with train_df
    df.polarity = df.polarity.replace(' negative', 'negative') # comes with test_dia_df
    # create aspect_polarity
    df['aspect_polarity'] = [':'.join([str(x), str(y)]) for x, y in zip(df['aspect'], df['polarity'])]
    return df

### additional load and prepare data functions for subtask C ###
def preproc_subtaskC(df, df_outer, cats, part_task):
    """ 
    pre-process data for subtask C1 if part_task == "aspect" and 
    for subtask C2 if part_task == "aspect_polarity".

    Args:
        df: dataframe
        outer_df: outer dataframe
        cats: categories (as list)
        part_task: "aspect" or "aspect_polarity"; defining the part of subtask C
    """
    # create nullmatrix
    matrix = np.zeros((len(df),len(cats)))

    opinions_index = pd.DataFrame(matrix, columns = cats)
    df_long = pd.concat([df, opinions_index], axis = 1)

    # fill opinions_index
    for i in np.arange(0, len(df_long)):
        for j in cats:
            if df_long[part_task][i] == j:
                df_long.loc[i,j] = 1

    # aggregate by id
    aggregations = {i:sum for i in cats}
    df_agg = df_long.groupby('id').agg(aggregations)

    # merge df_agg to df_outer (df without opinions)
    df_outer = pd.merge(df_outer, df_agg, on='id',  how='left')

    # convert to 0 / 1 labels
    for j in cats:
        df_outer[j] = df_outer[j].astype(bool)
        df_outer[j] = df_outer[j].astype(int)
        for i in np.arange(0, len(df_outer)):
            if df_outer.loc[i,j] > 0:
                df_outer.loc[i,j] = 1
    # delete irrelevant rows
    df_outer = df_outer.loc[df_outer.opinion, :]

    return df_outer

def get_cats(df_path, xml_filename = "train-2017-09-15.xml", part_task = "aspect"):
    """
    return full list of aspect/aspect+polarity categories as list.

    Args:
        df_path: path to data folder
        xml_filename: file name of XML dataset. Default is XML train dataset.
        part_task: "aspect" or "aspect_polarity"; defining part of Subtask C. 
    """

    tree = ET.parse(df_path+xml_filename)
    df = convert_df(tree)
    cats = []
    if part_task == "aspect":
        cats = df.aspect.unique()
        cats = np.delete(cats, 2) # delete nan
        cats = np.append(cats,'QR-Code')

    if part_task == "aspect_polarity":
        cats = df.aspect_polarity.unique()
        # add Gepäck:positive and all QR_Code combinations
        add = ['Gepäck:positive', 'QR-Code:negative', 'QR-Code:neutral', 'QR-Code:positive']
        cats = sorted(np.append(cats, add))
        # delete nan:nan
        cats = np.delete(cats, -1)

    return cats

### load and prepare functions for subtask D --> convert "from" and "to" sequence indices to BIO tags ###
def prep_df(df):
    """
    prepare data for subtask D.

    Arg:
        df: pre-processed dataframe
    """
    # drop NAs in opinion
    pdf = df.dropna(subset = ["target"])
    pdf = pdf[pdf.target != "NULL"]
    pdf[['from', 'to']] = pdf[['from', 'to']].astype(int)
    # if from > to, switch positions
    pdf.loc[pdf['to'] < pdf['from'], ['from', 'to']] = pdf.loc[pdf['to'] < pdf['from'], ['to', 'from']].values

    # create labels
    # define dictionary for categories and polarity
    aspect_dict = {
        'Allgemein': 'ALG', 
        'Atmosphäre': 'ATM', 
        'Informationen': 'INF',
        'DB_App_und_Website': 'APP',
        'Auslastung_und_Platzangebot': 'AUP',
        'Sonstige_Unregelmässigkeiten': 'SOU',
        'Zugfahrt': 'ZUG', 
        'Ticketkauf': 'TIC', 
        'Sicherheit': 'SIC',
        'Barrierefreiheit': 'BAF', 
        'Service_und_Kundenbetreuung': 'SUK', 
        'Connectivity': 'CON', 
        'Komfort_und_Ausstattung': 'KUA',
        'Toiletten': 'TOI', 
        'Gastronomisches_Angebot': 'GST', 
        'Image': 'IMG', 
        'Design': 'DSG', 
        'Reisen_mit_Kindern': 'RMK',
        'Gepäck': 'GEP',
        'QR-Code': 'QRC'
    }
    polarity_dict = {
        'neutral': 'NEU',
        'positive': 'POS',
        'negative': 'NEG'
    }
    pdf = pdf.replace({"aspect":aspect_dict, "polarity":polarity_dict})
    entities = pdf.aspect + ":" + pdf.polarity    

    pdf[['from', 'to']] = pdf[['from', 'to']].astype(int)
    return pdf, list(set(entities))

def transform_df(df):
    '''
    transform prepared data i.e. create list with tuples
    (text, {"target": (from, to, target, aspect, polarity)})

    Args: 
        df: dataframe
    '''
    tdf = []
    for i in df.id.unique():
        subset = df[df.id == i]
        entry = (subset.text.iloc[0], {"target": [(start, end, tar, asp, pol) for start, end, tar, asp, pol in zip(subset['from'], subset['to'], subset['target'], subset['aspect'], subset['polarity'])]})
        tdf.append(entry)
    return tdf

def bio_tagging_sentence(dataentry):
    """
    add BIO tags to documents.

    Arg:
        df: prepared dataframe entry as tuple of text and dictionary (see transform_df())
    """
    # get text of entry
    text_entry = dataentry[0]
    # get opinion entries
    entry = next(iter(dataentry[1].values()))
    # catch aspect + polarity
    asp_pol = [entry[t][3] + ':' + entry[t][4] for t in range(0, len(entry))]

    # split text in several parts: with and without opinion
    txts = []
    bio_tags = []
    # starting part
    startO_pos = entry[0][0]
    if startO_pos > 0:
        txts.append(text_entry[0:startO_pos])
        bio_tags.append('O')
    # middle part(s)
    BI_pos = [(entry[k][0], entry[k][1]) for k in range(0, len(entry))]
    O_pos = [(entry[k][1], entry[k+1][0]) for k in range(0, len(entry)-1)]
    # ending part
    endO_pos = (entry[0][1], len(text_entry))
    if endO_pos[0] != endO_pos[1]:
        O_pos.append(endO_pos)
    k = 0 # iterator for asp:pol
    for i, j in zip(BI_pos, O_pos):
        if i[0] != i[1]:
            a = text_entry[i[0]:i[1]]
            txts.append(a)
            bio_tags.append('I-' + asp_pol[k])
            k = k + 1
        if j[0] != j[1]:
            b = text_entry[j[0]:j[1]]
            txts.append(b)
            bio_tags.append('O')

    # exception: if O_pos is empty
    if not O_pos and len(BI_pos) == 1:
        a = text_entry[BI_pos[0][0]:BI_pos[0][1]]
        txts.append(a)
        bio_tags.append('I-' + asp_pol[0])

    #### extend BIO tags
    kk = 0
    bio_tags_full = []
    split = []
    for i, j in enumerate(bio_tags):
        txts_split = txts[i].split()
        n_subwords = len(txts_split)
        if j == 'O':
            O_tags = ['O'] * n_subwords
        else:
            O_tags = ['I-' + asp_pol[kk]] * n_subwords
            O_tags[0] = 'B-' + asp_pol[kk]
            kk = kk + 1
        bio_tags_full.append(O_tags)
        split.append(txts_split)
    return split, bio_tags_full

def bio_tagging_df(df):  
    """
    Return complete pre-processed dataframe of text and corresponding BIO tags.

    Args:
        df: dataframe
    """
    pdf, _ = prep_df(df)
    tdf = transform_df(pdf)
    bio_tags_df = [bio_tagging_sentence(l) for l in tdf]
    bio_tags_df = pd.DataFrame(bio_tags_df, columns = ["text", "bio_tags"])
    return bio_tags_df


def sample_to_tsv(df_path, xml_filename, save_as_tsv=True):
    """
    pre-process data and save as TSV file.

    Args:
      df_path: location of dataframe
      xml_filename: name of file in XML format
      save_as_tsv: flag for saving as TSV file
    """
    
    print("Original XML Dateframe: ", xml_filename)
    tree = ET.parse(df_path+xml_filename)
    root = tree.getroot()

    ################## data for subtask A + B ########################
    df = create_outer_df(root)

    ################## data for subtask C + D ########################
    df_op = convert_df(tree)
    df_op = df_op.dropna(subset = ["text"])    

    ################## data for subtask C ############################
    # create categories array (same for all data!)
    cats = get_cats(df_path, part_task = "aspect")
    df_cat = preproc_subtaskC(df_op, df, cats, 'aspect')

    cats_pol = get_cats(df_path, part_task = "aspect_polarity")
    df_cat_pol = preproc_subtaskC(df_op, df, cats_pol, 'aspect_polarity')

    if save_as_tsv:
        df_type = xml_filename.split("-")[0]
        # for subtask A + B
        df.to_csv(df_path+df_type+"_df.tsv", sep="\t", index = False, header = True)
        # for subtask C
        df_cat.to_csv(df_path+df_type+"_df_cat.tsv", sep="\t", index = False, header = True)
        df_cat_pol.to_csv(df_path+df_type+"_df_cat_pol.tsv", sep="\t", index = False, header = True)
        # for subtask D (without BIO tags)
        df_op.to_csv(df_path+df_type+"_df_opinion.tsv", sep="\t", index = False, header = True)


def main():
    """
    pre-process and save full data: train, dev, test_syn, test_dia.
    will save 16 (4x4) TSV files in data folder (see sample_to_tsv()).
    """

    df_path = "./data/"
    sample_to_tsv(df_path, "train-2017-09-15.xml")
    sample_to_tsv(df_path, "dev-2017-09-15.xml")
    sample_to_tsv(df_path, "test_syn-2017-09-15.xml")
    sample_to_tsv(df_path, "test_dia-2017-09-15.xml")
    print("Complete data is saved as TSV files in ", df_path)

if __name__ == "__main__":
    main()