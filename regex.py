# check if a text contains punctuation

import re
def check_punctuation (text):
  result = re.search(r"[^a-zA-Z ]", text)
  return result != None

print(check_punctuation("This is a sentence that ends with a period.")) # True
print(check_punctuation("This is a sentence fragment without a period")) # False
print(check_punctuation("Aren't regular expressions awesome?")) # True
print(check_punctuation("Wow! We're really picking up some steam now!")) # True
print(check_punctuation("End of the line")) # False


print(re.search(r"Py.*n", "Pygmalion")) # <re.Match object; span=(0, 9), match='Pygmalion'>
print(re.search(r"Py.*n", "Python Programmin")) 
# <re.Match object; span=(0, 17), match='Python Programmin'>
# we say that the process is greedy!!
# to make it less greedy, use character classes
print(re.search(r"Py[a-z]*n", "Python Programmin")) 
# <re.Match object; span=(0, 6), match='Python'>

print(re.search(r"o+l+", "goldfish"))
# <re.Match object; span=(1, 3), match='ol'>

print(re.search(r"o+l+", "woolly"))

# matches a in a word at least 2 times
def repeating_letter_a(text):
    result = re.search(r"[aA].*a", text)
    return result != None

print(repeating_letter_a("banana")) # True
print(repeating_letter_a("pineapple")) # False
print(repeating_letter_a("Animal Kingdom")) # True
print(repeating_letter_a("A is for apple")) # True


# matches dates like dd.mm.aaaa in any text

def date_matches(date):
    return re.search(r"[0-9][0-9]\.[0-9][0-9]\.[0-9][0-9][0-9][0-9]", date)

print(date_matches(r"15.09.1998"))
# <re.Match object; span=(0, 10), match='15.09.1998'>

print(date_matches(r"315.09.1998"))
# <re.Match object; span=(1, 11), match='15.09.1998'>


def date_matches_two(date):
    return re.search(r"[0-9]+\.[0-9]+\.[0-9]+", date)

print(date_matches_two(r"15.09.1998"))
# <re.Match object; span=(0, 10), match='15.09.1998'>

print(date_matches_two(r"315.09.1998"))
# <re.Match object; span=(0, 11), match='315.09.1998'>


# optinal research with ?

print(re.search(r"p?on", "python"))
# <re.Match object; span=(4, 6), match='on'>


print(re.search(r"p?r", " I like python programming"))
# <re.Match object; span=(15, 17), match='pr'>

def date_matches_three(date):
    pattern = '\d{2}\.\d{2}\.\d{4}'
    return re.findall(pattern, date)

print(date_matches_three(r"15.09.1998"))
# <re.Match object; span=(0, 10), match='15.09.1998'>

# open the file where the text has been extracted and get all the date that are 
# in the form dd.mm.aaaa

with open('./recognized.txt', 'r') as file:
    info = file.read().rstrip('\n')
    
print(date_matches_three(info))
