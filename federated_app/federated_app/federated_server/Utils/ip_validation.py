from logging import exception


class IPAddress(object):

    @staticmethod
    def validReturnIPAddress(IP):
        """
        :type IP: str
        :rtype: str
        """
        def isIPv4(s):
            try:
                return str(int(s)) == s and 0 <= int(s) <= 255
            except Exception as ex:
                raise Exception(f"Ipv4 address is invalid, {ex}")

        def isIPv6(s):
            if len(s) > 4:
                return False
            try:
                return int(s, 16) >= 0 and s[0] != '-'
            except Exception as ex:
                return Exception(f"Ipv6 address is invalid, {ex}")

        if IP.count(".") == 3 and all(isIPv4(i) for i in IP.split(".")):
            return str(IP)
        elif IP.count(":") == 7 and all(isIPv6(i) for i in IP.split(":")):
            return str(IP)
        elif str(IP) == "[::]:8999":
            return str(IP)

        raise Exception("Invalid ip address")
