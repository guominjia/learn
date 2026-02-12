from scrapy.core.downloader.handlers.http11 import HTTP11DownloadHandler
from scrapy.http import Response
from twisted.internet import defer
import requests
from requests_kerberos import HTTPKerberosAuth, OPTIONAL

class KerberosHTTPDownloadHandler(HTTP11DownloadHandler):    
    def download_request(self, request, spider):
        return self._download_with_requests(request)

        return super().download_request(request)
    
    @defer.inlineCallbacks
    def _download_with_requests(self, request):
        try:
            session = requests.Session()
            session.auth = HTTPKerberosAuth(mutual_authentication=OPTIONAL)
            
            response = yield self._make_request(session, request)
            
            scrapy_response = Response(
                url=str(response.url),
                status=response.status_code,
                headers=response.headers,
                body=response.content,
                request=request,
            )
            
            defer.returnValue(scrapy_response)
        
        except Exception as e:
            request.spider.logger.error(f"Kerberos download failed for {request.url}: {e}")
            raise
    
    def _make_request(self, session, scrapy_request):
        from twisted.internet import threads
        
        def _blocking_request():
            return session.request(
                method=scrapy_request.method,
                url=scrapy_request.url,
                headers=dict(scrapy_request.headers.to_unicode_dict()),
                cookies=scrapy_request.cookies,
                data=scrapy_request.body,
                timeout=30,
            )
        
        return threads.deferToThread(_blocking_request)