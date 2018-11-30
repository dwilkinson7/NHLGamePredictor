import csv
import getopt
import json
import sys
import urllib.request

def main(argv):
	first = True

# Gathering Oilers Game stats since 2005
	with urllib.request.urlopen('http://www.nhl.com/stats/rest/team?isAggregate=false&reportType=basic&isGame=true&reportName=teamsummary&sort=[{"property":"gameDate","direction":"ASC"}]&factCayenneExp=gamesPlayed>=1&cayenneExp=gameDate>="2005-09-01"%20and%20gameDate<="2018-11-29"%20and%20gameTypeId=2%20and%20teamId=22') as url:
		data = json.loads(url.read().decode())
	
	with open('NHL_Game_Stats_Train.csv', 'a') as csvfile:
		if first:
			fieldnames = []
			for field in data['data'][0]:
				fieldnames.append(field)
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames, lineterminator='\n')
		if first:
			writer.writeheader()
			first = False
		for row in data['data']:
			writer.writerow(row)
				
if __name__ == "__main__":
   main(sys.argv[1:])