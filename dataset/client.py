import requests

print "Downloading modules from cloud."
# svrurl  = 'http://{}/train'.format(args.svr)
url = 'http://10.193.20.144:5000/dataset'
headers = {"Content-type":"application/json","Accept": "application/json"}
r = requests.get(url, headers=headers)
if(200 == r.status_code):
   f = open('./feature.tgz','w')
   f.write(r.content)
   f.close()
else:
    print("Receive unexpected status code {}".format(r.status_code))
