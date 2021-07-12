
from urllib.parse import urlparse, parse_qs
from http.server import BaseHTTPRequestHandler, HTTPServer

from bert import bert
hostName = "localhost"
import sys
try:
    serverPort = int(sys.argv[1]) if len(sys.argv) == 2 else 8080
except Exception as e:
    serverPort=8080
A = bert()

class MyServer(BaseHTTPRequestHandler):
    def do_GET(self):
        if (self.path[:5] == "/api/"):
            self.do_reponse_GET_API()
        else:
            try:
                with open('index.html', 'rb') as f:
                    self.add_parameter('text/html')
                    self.wfile.write(f.read())

                return
            except IOError:
                self.send_error(404, 'File Not Found: % s' % self.path)

    def do_reponse_GET_API(self):
        print(self.path)
        parsed_url = urlparse(self.path)
        query = parse_qs(parsed_url.query)
        print(query, type(query))
        self.add_parameter("application/json")
        if ("sentence" in query):
          
            print(query["sentence"][0])
            self.wfile.write(
                bytes('{"result":"'+str(A.classify_sentiment(query["sentence"][0])[0][0])+'"}', "utf-8"))

        else:
            self.wfile.write(
                bytes('{err:"sentence is required"}', "utf-8"))

    def add_parameter(self, arg0):
        self.send_response(200)
        self.send_header("Content-type", arg0)
        self.end_headers()


if __name__ == "__main__":
    webServer = HTTPServer((hostName, serverPort), MyServer)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")
