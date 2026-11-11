import re
from urllib.parse import urlsplit

SHORTENERS = {
    "bit.ly", "goo.gl", "tinyurl.com", "ow.ly", "t.co", "is.gd", "buff.ly", "adf.ly", "bit.do",
    "cutt.ly", "shorte.st", "tiny.cc", "lnkd.in", "db.tt", "qr.ae", "rebrand.ly", "v.gd", "x.co",
    "tr.im", "cli.gs", "migre.me", "ff.im", "url4.eu", "twit.ac", "su.pr", "twurl.nl", "snipurl.com",
    "short.to", "budurl.com", "ping.fm", "post.ly", "just.as", "bkite.com", "snipr.com", "fic.kr",
    "loopt.us", "doiop.com", "kl.am", "wp.me", "rubyurl.com", "om.ly", "to.ly", "yfrog.com",
    "u.to", "j.mp", "bc.vc", "s.id", "shorturl.at", "rb.gy", "soo.gd", "clck.ru", "yourls.org",
    "1url.com", "tweez.me", "v.ht", "link.zip.net", "x.gd", "t.ly", "go2l.ink", "qrco.de",
}

IPV4 = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
HEX_IP = re.compile(r"^(0x[0-9a-f]{1,2}\.){3}0x[0-9a-f]{1,2}$|^0x[0-9a-f]{8}$")
SECOND_LEVEL = {"co", "com", "ac", "gov", "org", "net", "edu", "gob", "nic", "or", "ne", "go"}


def host_of(url):
    if "://" not in url:
        url = "http://" + url
    try:
        host = urlsplit(url).hostname or ""
    except ValueError:
        host = ""
    return host.lower()


def registered_domain(url):
    labels = [p for p in host_of(url).split(".") if p]
    if labels and labels[0] == "www":
        labels = labels[1:]
    if IPV4.match(".".join(labels)):
        return ".".join(labels)
    if len(labels) >= 3 and len(labels[-1]) == 2 and labels[-2] in SECOND_LEVEL:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def extract(url):
    host = host_of(url)
    bare = host[4:] if host.startswith("www.") else host

    using_ip = -1 if (IPV4.match(host) or HEX_IP.match(host) or host.startswith("[")) else 1

    n = len(url)
    long_url = 1 if n < 54 else (0 if n <= 75 else -1)

    short_url = -1 if (bare in SHORTENERS or host in SHORTENERS) else 1

    symbol_at = -1 if "@" in url else 1

    redirecting = -1 if url.rfind("//") > 7 else 1

    prefix_suffix = -1 if "-" in bare else 1

    labels = [p for p in bare.split(".") if p]
    if labels and len(labels[-1]) == 2 and not IPV4.match(bare):
        labels = labels[:-1]
    dots = max(len(labels) - 1, 0)
    sub_domains = 1 if dots <= 1 else (0 if dots == 2 else -1)

    https_domain = -1 if "https" in host else 1

    return {
        "UsingIP": using_ip,
        "LongURL": long_url,
        "ShortURL": short_url,
        "Symbol@": symbol_at,
        "Redirecting//": redirecting,
        "PrefixSuffix-": prefix_suffix,
        "SubDomains": sub_domains,
        "HTTPSDomainURL": https_domain,
    }
