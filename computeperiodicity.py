import time
import urllib2
import json
import re
import numpy as np
import numpy.fft as fft
import requests
import pandas  as pd
import matplotlib.pyplot as plt
from datetime import datetime
from elasticsearch import Elasticsearch

globalinterval="10m"

################################################################################
# Load Data From ES
################################################################################
def loadDataFromES(begin,end):
    print "# Load Data From ES"
    es = Elasticsearch(hosts=['ictcs.be:8093'])
    request={
          "query": {
            "filtered": {
              "query": {
                "query_string": {
                  "query": "*",
                  "analyze_wildcard": True
                }
              },
              "filter": {
                "bool": {
                  "must": [
                    {
                      "range": {
                        "generationtime": {
                          "gte": begin,
                          "lte": end,
                          "format": "epoch_millis"
                        }
                      }
                    }
                  ],
                  "must_not": []
                }
              }
            }
          },
          "aggs": {
            "time": {
              "date_histogram": {
                "field": "generationtime",
                "interval": globalinterval,
                "time_zone": "Europe/Berlin",
                "min_doc_count": 1,
                "extended_bounds": {
                  "min": begin,
                  "max": end
                }
              },
              "aggs": {
                "area": {
                  "terms": {
                    "field": "area",
                    "size": 5,
                    "order": {
                      "maxvalue": "desc"
                    }
                  },
                  "aggs": {
                    "maxvalue": {
                      "max": {
                        "field": "value"
                      }
                    }
                  }
                }
              }
            }
          }
        }

    res = es.search(
      index = 'energy*',
      doc_type = 'conso',
      size = 0,
      body = request)

    #print res;
    print json.dumps(res);

    mylist=res['aggregations']['time']['buckets']
    series={}
    indexlist=[];

    for i,val in enumerate(mylist):
        indexlist.append(datetime.fromtimestamp(val['key']/1000))
        for j,val2 in enumerate(val['area']['buckets']):
            if((val2['key'] not in series)):
                series[val2['key']]=[];

            series[val2['key']].append(val2['maxvalue']['value'])

    series=pd.DataFrame(series,index=indexlist);

    #plt.plot(series)
    #plt.show()

    print "*"*10
    print series.shape
    print "*"*10
    print series.describe()

    return series;

################################################################################
# Compute the time serie periodicity
################################################################################

def computePeriodicity(data,begin,end):
    time_window = (end-begin)/1000

    Period=60*60*48;
    N = len(data) #datapoints
    T = (Period*N)/time_window #Cycles per minute

    print "*"*90
    print "Time Window:%d s" %(time_window)
    print "Samples:%d" %(N)
    print "Frequency per %d seconds:%.4f" %(Period,T)

    a=np.abs(fft.rfft(data, n=data.size))[1:]
    freqs = fft.rfftfreq(data.size, d=1./T)[1:]
    freqs = np.divide(Period,freqs)

    max_freq = freqs[np.argmax(a)]
    print "Full max found at %s second period (%s minutes) hours (%.2f)" % (max_freq, max_freq/60,max_freq/3600)


    biggestind = np.argpartition(a, -5)[-5:]
    print biggestind
    for i,val in enumerate(biggestind):
        max_freq=freqs[val]
        print "- Max Peaks (%d) found at %s second period (%s minutes) hours (%.2f)" % (a[val],max_freq, max_freq/60,max_freq/3600)

    plt.subplot(211,axisbg='black')
    plt.bar(freqs,a,edgecolor="gray",linewidth=2)
    plt.plot(freqs,a, 'r--')
    plt.grid(b=True, color='w')

    plt.subplot(212,axisbg='black')
    plt.plot(np.linspace(0,time_window,data.size),data,'g')
    plt.grid(b=True,axis="y", color='w')

    plt.show()

    #for i in range(0,data.size):
    #    print "%d=>%d = %d" %(i,data[i],serieTRIB_DEP2asArray[i])



################################################################################

#2 days
#begin=1479302502080
#end=1479475302080

#7 days
#begin=1478869373282
#end=1479474173282

#14 days
begin=1478259284174
end=1479468884174

print "*"*90
print "Time=%d s" %((end-begin)/(1000))
print "Time=%d mn" %((end-begin)/(1000*60))
print "Time=%d hours" %((end-begin)/(1000*60*60))
print "Time=%d days" %((end-begin)/(1000*60*60*24))
print "*"*90

res=loadDataFromES(begin,end)

npseries=res.values;
vals0=npseries[:,1]
#vals0=npseries[:,1]+np.random.uniform(low=0, high=100, size=len(vals0))
absicess = np.linspace(0,1,len(vals0))
vals0=npseries[:,1]+200*np.cos(20*np.pi*absicess)+200*np.cos(40*np.pi*absicess)
print len(absicess)
print len(vals0)

computePeriodicity(vals0,begin,end)
