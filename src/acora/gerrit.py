from requests.auth import HTTPDigestAuth
from pygerrit2 import GerritRestAPI, HTTPBasicAuth
from dateutil import parser
import requests
from pprint import pprint
import urllib
import logging
import traceback
import time

class GerritReviewDataDownloader(object):
    """Allows downloading code and comments from Gerrit instances."""

    def __init__(self, base_url, auth=None, sleep_between_pages=0):
        """
        Parameters
        ----------
        base_url : str
            The base URL address of the Gerrit instance to download data from.
        auth : pygerrit2.HTTPBasicAuth, optional
            The data used for authentication (e.g., pygerrit2.HTTPBasicAuth('username', 'password'), 
            None means a guest access (default is None).
        sleep_between_pages : int, optional
            Pause time between each page being fetched measured in seconds (default is 0)
        """
        self.logger = logging.getLogger('acora.data.gerrit.GerritReviewDataDownloader')
        self.base_url = base_url
        self.auth = auth
        self.client =  GerritRestAPI(url=self.base_url, auth = self.auth)
        self.sleep_between_pages = sleep_between_pages


    def download_commented_lines_to_csv(self, 
            filename, 
            base_query="/changes/?q=status:merged&o=ALL_FILES&o=ALL_REVISIONS&o=DETAILED_LABELS", 
            max_queries=1000, 
            n=500,
            max_fails=10,
            sep=';',
            from_date=None, 
            to_date=None):
        """ Processes changes and extracts lines that were commented on and save them to a csv file.

        Parameters
        ----------
        filename : str
            A path to an output csv file.
        base_query : str
            A query used to obtain changes from the Gerrit instance.
        max_queries : int
            A maximum number of queries to be made.
        n : int
            A number of changes requested in each batch from the Gerrit instance.
        max_fails : int
            A number of fails that will be ignored. The process will finish if the number is reached.
        sep : str
            A symbol that is used to separate columns in the csv file.
        from_date : str or None
            If provided, only not older changes will be returned.
        to_date : str or None
            If provided, only not newer changes will be returned.
        """

        with open(filename, 'w', encoding='utf-8', errors='ingore') as out:
            out.write(f'change_id{sep}date{sep}revision_id{sep}filename{sep}line{sep}start_line{sep}end_line{sep}line_contents{sep}message\n')

            start = 0
            has_more = True
            fails = 0
            queries = 0

            if from_date:
                from_date = parser.parse(from_date)
            if to_date:
                to_date = parser.parse(to_date)

            while has_more and queries < max_queries and fails < max_fails:
                try:
                    query = f'{base_query}&start={start}&n={n}'
                    self.logger.debug(f'Query: {query}')
                    changes = self.client.get(query, headers={'Content-Type': 'application/json'})
                    queries += 1
                    number_of_changes = len(changes)
                    self.logger.debug(f'Received {number_of_changes} changes')

                    # here we process the changes
                    for idx, change in enumerate(changes, start=1):
                        change_id = change['id']
                        date_created_str = change['created']
                        date_created = parser.parse(date_created_str)
                        
                        if from_date and from_date >= date_created:
                            continue

                        if to_date and to_date <= date_created:
                            continue
                            
                        if idx % 100 == 0:
                            self.logger.info(f'Extracting change: {idx} of {number_of_changes} starting at : {start}') 
                        
                        revisions = change['revisions']

                        for rev_id in list(revisions.keys()):
                            current_comment = self.client.get(f'/changes/{change_id}/revisions/{rev_id}/comments', 
                                                                headers={'Content-Type': 'application/json'})
                            
                            # not all revisions have comments, so we only look for those that have them
                            if len(current_comment) > 0:                
                                for one_file, one_comment in current_comment.items():                    
                                    try:
                                        # this code extracts information about the comment 
                                        # things like which file and which lines
                                        for one_comment_item in one_comment:
                                            file_str = one_file
                                            
                                            # a few if-s because not always all parameters are there
                                            if 'line' in one_comment_item:
                                                line_str = one_comment_item['line']
                                            else:
                                                line_str = ''

                                            if 'message' in one_comment_item:
                                                message_str = one_comment_item['message']
                                            else:
                                                message_str = ''
                                            
                                            # if there is a specific line and characters as comments
                                            if 'range' in one_comment_item:
                                                start_line_str = one_comment_item['range']['start_line']
                                                end_line_str = one_comment_item['range']['end_line']                      
                                            else:                            
                                                start_line_str = '0'
                                                end_line_str = '0'

                                            # if we can extract something from a file
                                            # then here is where we do it
                                            if line_str != '':
                                                # we need the line below to properly encode the filename as URL
                                                url_file_id = urllib.parse.quote_plus(file_str)
                                                file_content_string = f'/changes/{change_id}/revisions/{rev_id}/files/{url_file_id}/content'
                                                file_contents = self.client.get(file_content_string, headers={'Content-Type': 'application/json'})
                                                file_lines = file_contents.split("\n")

                                                # if we have the lines delimitations (comment that is linked to lines)
                                                if start_line_str != '0':
                                                    start_line = int(start_line_str) - 1
                                                    if end_line_str != '0':
                                                        end_line = int(end_line_str) - 1  
                                                    else: 
                                                        end_line = len(file_lines) - 1

                                                    for one_line in file_lines[start_line:end_line]:
                                                        str_to_csv = str(change_id) + sep + \
                                                        date_created_str + sep + \
                                                        str(rev_id) + sep + \
                                                        file_str + sep + \
                                                        str(line_str) + sep + \
                                                        str(start_line_str) + sep + \
                                                        str(end_line_str) + sep + \
                                                        one_line.replace("\n", " _ ").replace('\r', '_').replace(sep, '_') + sep + \
                                                        message_str.replace("\n", " _ ").replace('\r', '_').replace(sep, '_')
                                                        out.write(str_to_csv + "\n")
                                                elif int(line_str) < len(file_lines):                                
                                                    # and if there are no delimitation, but there is a starting line
                                                    # and the starting line is below the end of the file
                                                    one_line = file_lines[int(line_str)-1]
                                                    str_to_csv = str(change_id) + sep + \
                                                            date_created_str + sep + \
                                                            str(rev_id) + sep + \
                                                            file_str + sep + \
                                                            str(line_str) + sep + \
                                                            str(start_line_str) + sep + \
                                                            str(end_line_str) + sep + \
                                                            one_line.replace("\n", " _ ").replace('\r', '_').replace(sep, '_') + sep + \
                                                            message_str.replace("\n", " _ ").replace('\r', '_').replace(sep, '_')
                                                    out.write(str_to_csv + "\n")
                                            else: 
                                                # there is no line specified, then we take the comment for the entire file
                                                for one_line in file_lines:
                                                        str_to_csv = str(change_id) + sep + \
                                                        date_created_str + sep + \
                                                        str(rev_id) + sep + \
                                                        file_str + sep + \
                                                        str(line_str) + sep + \
                                                        str(start_line_str) + sep + \
                                                        str(end_line_str) + sep + \
                                                        one_line.replace("\n", " _ ").replace('\r', '_').replace(sep, '_') + sep + \
                                                        message_str.replace("\n", " _ ").replace('\r', '_').replace(sep, '_')
                                                        out.write(str_to_csv + "\n")

                                    except:
                                        # this is a brutal exception handling, but we cannot check for all problems
                                        self.logger.info('Unhandled exception, moving on')
                                        self.logger.error(traceback.format_exc())
                                        self.logger.error(one_comment)
                    
                
                    if has_more:
                        start += n
                except:
                    fails += 1
                    self.logger.error(traceback.format_exc())
                    time_sleep = self.sleep_between_pages
                    self.logger.info(f'Waiting for {time_sleep} seconds because of HTTP error...')
                    time.sleep(time_sleep)


    def download_lines_to_csv(self, 
            filename, 
            base_query="/changes/?q=status:merged&o=ALL_FILES&o=ALL_REVISIONS&o=DETAILED_LABELS", 
            max_queries=1000, 
            n=500,
            max_fails=10,
            sep=';',
            from_date=None, 
            to_date=None):
        """ Processes changes and extracts lines and save them to a csv file.

        Parameters
        ----------
        filename : str
            A path to an output csv file.
        base_query : str
            A query used to obtain changes from the Gerrit instance.
        max_queries : int
            A maximum number of queries to be made.
        n : int
            A number of changes requested in each batch from the Gerrit instance.
        max_fails : int
            A number of fails that will be ignored. The process will finish if the number is reached.
        sep : str
            A symbol that is used to separate columns in the csv file.
        from_date : str or None
            If provided, only not older changes will be returned.
        to_date : str or None
            If provided, only not newer changes will be returned.
        """

        with open(filename, 'w', encoding='utf-8', errors='ingore') as out:
            out.write(f'change_id{sep}date{sep}revision_id{sep}filename{sep}line_contents\n')

            start = 0
            has_more = True
            fails = 0
            queries = 0

            if from_date:
                from_date = parser.parse(from_date)
            if to_date:
                to_date = parser.parse(to_date)

            while has_more and queries < max_queries and fails < max_fails:
                try:
                    query = f'{base_query}&start={start}&n={n}'
                    self.logger.debug(f'Query: {query}')
                    changes = self.client.get(query, headers={'Content-Type': 'application/json'})
                    queries += 1
                    number_of_changes = len(changes)
                    self.logger.debug(f'Received {number_of_changes} changes')

                    # here we process the changes
                    for idx, change in enumerate(changes, start=1):
                        change_id = change['id']
                        date_created_str = change['created']
                        date_created = parser.parse(date_created_str)
                        
                        if from_date and from_date >= date_created:
                            continue

                        if to_date and to_date <= date_created:
                            continue
                            
                        if idx % 100 == 0:
                            self.logger.info(f'Extracting change: {idx} of {number_of_changes} starting at : {start}') 
                        
                        revisions = change['revisions']

                        for rev_id in list(revisions.keys()):
                            # first we get all the files from the revision
                            files_query = f'/changes/{change_id}/revisions/{rev_id}/files/'
                            try:
                                files = self.client.get(files_query, headers={'Content-Type': 'application/json'})

                                # then we go through all of these files
                                for file_id in list(files.keys()):
                                    # for now, we only filter out the commit message as we do not really need it
                                    # but we can also filter out other types of files
                                    # fileID is the name of the file, so we can use that to filter 
                                    if 'COMMIT' not in file_id:
                                        # first we extract the diff structure for the file in this revisio
                                        url_file_id = urllib.parse.quote_plus(file_id)
                                        diff_query = f'/changes/{change_id}/revisions/{rev_id}/files/{url_file_id}/diff'
                                        try:
                                            diff = self.client.get(diff_query, headers={'Content-Type': 'application/json'})

                                            # here we extract the content of the diff, 
                                            # which is the lines and commit messages
                                            content = diff['content']


                                            # here we go through all churns of the change
                                            # and based on the type we either include or exclude them
                                            # in the resulting file
                                            # the content has many churns, so we need to extract them and process one-by-one
                                            for churn in content:
                                                # the first element of the keys is the type - a, b, or ab
                                                # a - lines removed
                                                # b - lines added
                                                # ab - lines that are the same in both revisions
                                                churn_type = list(churn.keys())[0]

                                                # here, we check if the type is b, which means lines added
                                                # b denotes that the lines are in revision b, but not in revision a
                                                # so, this means that these lines are added
                                                if churn_type == 'b':
                                                    for lines in list(churn.values()):
                                                        # in this loop we take all the lines, 
                                                        # but in principle we can filter some lines here
                                                        # then we need to add an if inside this loop
                                                        for line in lines:
                                                            try:
                                                                line_contents = line.replace("\n", " _ ").replace('\r', '_').replace(sep, '_')
                                                                out.write(f'{change_id}{sep}{date_created_str}{sep}{rev_id}{sep}{file_id}{sep}{line_contents}\n')
                                                            except UnicodeEncodeError:
                                                                self.logger.info("Encoding problem, skipping one line")  
                                        except:
                                            self.logger.debug(f"Failed to query: {diff_query}")  
                            except:
                                self.logger.debug(f"Failed to query: {files_query}")                            
            
                    if has_more:
                        start += n
                except:
                    fails += 1
                    self.logger.error(traceback.format_exc())
                    time_sleep = self.sleep_between_pages
                    self.logger.info(f'Waiting for {time_sleep} seconds because of HTTP error...')
                    time.sleep(time_sleep)









    



