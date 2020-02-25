from six.moves import configparser
config = configparser.ConfigParser()
config.read("parameters.ini")

def get(section,param):
    return config.get(section,param)

#print(get("TestDB", "host"))