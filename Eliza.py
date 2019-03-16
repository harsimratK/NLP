r'''
@ Description:  The first assignment is to write an Eliza program in Python. The program is called eliza.p
                
                Eliza is a program that plays the role of a  psychotherapist.
                It is able to get engage in a conversation with user.q
                It recognize certain key words and respond simply based on that word being present in the input. 
                It also transform certain simple sentence forms from input into questions				
@Usage Instructions:  when you run the prog eliza will greet and ask for your name. Then from user input it will  get the name using regular expressions.
					Once it get the nmae it will ask how eliz can help.User will give its input and from the given answer Eliza will form questions to continue the  conversation. If eliza is not able to get useful information, it will ask user to elaborate or tell more.
					Example:
					PS C:\Users\savi0\Ex1> python eliza.py
					[Eliza] I am Eliza,I am a Psychotherapist and you are?
					Hi eliza I am savi
					[Eliza] How are you doing today,savi?
					savi >I want chocolate.
					[Eliza] Why do you thnk you want chocolate?
					savi >I sgfdxghasghx
					[Eliza] Hmmm! Can you elaborate savi.					
					To exit eliza user can say "Bye"					
@Algorithim:  Once user gives an input, program run regex to get useful information.
              Based on information it decides the state from which regex belongs.Program has a question library based on state.
              Once it decides the state it ask question from STATE_Q_LIBRARY.and user is prompted for an answers.Again prog tries to find useful info and  STATE_TRANSITION_TABLE is used to decide the state from which next ques will be asked.                  
@ Author: Sri Ram Sagar Kappagantula,
          Harsimrat Kaur and
          Ritika De.
@ Date: September 17, 2018

'''
import re
import logging
from random import randint
from random import choice

# Questions library based on state.
STATE_Q_LIBRARY = {'GREET': {1: '[{0}] Hi! I am {0}! I am a Psychotherapist.', 2: '[{0}] Hello! This is {0}! I am a Psychotherapist', 3: '[{0}] I am {0},I am a Psychotherapist and you are?'},
				   'HELP':{1:'[{0}]How can I help you today,{1}?', 2:'[{0}] How are you doing today,{1}?', 3:'[{0}]Is there anything I can help you with today,{1}?'},
                   'WANT': {1:'[{0}]Why do you think you want {1}?',2:'[{0}]Do you really need {1}?',3:'[{0}]How will you feel if you get{1}?'},
                   'FEEL': {1:'[{0}]What made you feel {1}',2:'[{0}]Do you enjoy feeling {1}',3:'[{0}]For how long have you been feeling {1}'},
                   'HAVE': {1:'[{0}] Do you feel happy having {1}?',2:'[{0}]How will you feel if you lost your {1}?',3:'[{0}] would you like sharing your {1}?'},
				   'DID':{1:'[{0}] Does it please you doing {1}?',2:'[{0}]Can you elaborate your process for doing {1}.',3:'[{0}] what made you do{1}?'},
                   'CONFUSED': {1:'[{0}] Hmmm! Can you elaborate {1}.', 2:'[{0}] Tell me more! {1}', 3:'[{0}] I did not understand what you said. {1}!'},
                   'EXIT': {1: '[{0}] Bye! {1}', 2: '[{0}] Have a good day! {1}', 3: '[{0}] Enjoy your rest of the day! {1}'}
                  }

# Regex library used to fetch information from user responses based on state.
STATE_I_LIBRARY = {'GREET': r'(AM | am | is | IS)\s*(?P<response>[a-zA-Z0-9]+).*$',
                   'WANT': r'(need| want |crave| NEED | WANT | CRAVE)\s+(?P<response>.*).*$',
                   'FEEL': r'(feel | FEEL | Feel | feel)(ings|ing)?\s+(?P<response>.*).*$',
                   'HAVE': r'(have | had | HAVE | HAD)\s+(?P<response>.*).*$',
                   'DID': r'(think | taught | mind)\s+(?P<response>.*).*$'
                   }
            
# Converstion state transition table based on the ability of machine to fetch info from user responses.
STATE_TRANSITION_TABLE = {('GREET', 'CONFUSED'):'GREET', ('GREET', 'INFO'): 'HELP',
                          ('HELP', 'CONFUSED'): 'HELP',  ('HELP', 'INFO'): 'DECIDE_STATE',
                          ('WANT', 'CONFUSED'): 'WANT',  ('WANT', 'INFO'): 'DECIDE_STATE',
                          ('FEEL', 'CONFUSED'): 'FEEL',  ('FEEL', 'INFO'): 'DECIDE_STATE',
                          ('HAVE', 'CONFUSED'): 'HAVE',  ('HAVE', 'INFO'): 'DECIDE_STATE',
                          ('CONFUSED','INFO'): 'DECIDE_STATE'
                         }

class Machine(object):
    ''' Bot Machine which traverses through the conversations states and uses necessary state infromation 
    to process and respond to the user.
    '''

    def __init__(self, agent_name, current_state):
        self.current_state = current_state
        self.previous_state = None
        self.__low = 1
        self.__high = 3
        self.machine_name = agent_name
        self.user_name = None
        self.all_state_response = None
    
    def ask_question(self, current_state, args):
        '''Generates a question'''
        question = STATE_Q_LIBRARY.get(current_state).get(randint(self.__low, self.__high))
        return question.format(*args)

    def get_response(self):
       '''Prompt user to type something.'''
       string = input(self.user_name + ' >') if self.user_name else input() 
       return string.strip()
    
    def classify(self, statement):
        '''Cassifies the responses of the user is infromative or confusing.'''
        if statement == None:
            return 'CONFUSED'
        else:
            return 'INFO'

    def check_exit(self, response, next_state):
        ''' Check to transition to exit or to next dialogue.'''
        if any(x in response.upper() for x in ['BYE', 'EXIT', 'QUIT', 'GOOD NIGHT']):
            return 'EXIT'
        return next_state

    def run_all_regex(self, input_string):
        ''' Regex runner for all states.'''
        response = None
        results = {}
        for state, regex in STATE_I_LIBRARY.items():
            re_obj = re.search(regex, input_string)
            try:
                response = re_obj.group('response')
            except:
                continue
            _class = self.classify(response)
            results[state] = (response, _class)
        return results
        
    def list_state_with_info(self):
        '''List out all infromative state by checking the response classification.'''
        return {state : item for state, item in self.all_state_response.items() if item[1] == 'INFO'}

    def decide_state(self):
        ''' Decides which conversation state the program should go to.'''
        if hasattr(self.all_state_response, 'GREET'): 
            self.all_state_response.pop('GREET')
        return choice(list(self.list_state_with_info().keys())) if len(self.list_state_with_info()) else 'CONFUSED'

    def run(self):
        " Run methodology where the machine computes and makes state jumps using state information and response classification"
        input_string = None
        user_response = None
        while(True):
            if self.current_state == 'GREET':
                next_state = 'CONFUSED'
                print(self.ask_question(self.current_state, [self.machine_name]))
                input_string = self.get_response()
                self.all_state_response = self.run_all_regex(input_string)
                item = self.all_state_response.get(self.current_state)
                if item:
                    response, _class = item
                    next_state = STATE_TRANSITION_TABLE.get((self.current_state, _class))
                next_state = self.check_exit(input_string, next_state)
                if next_state == 'HELP':
                    self.user_name = response
                self.previous_state = self.current_state
                self.current_state = next_state
                continue

            elif self.current_state == 'HELP':
                self.all_state_response = {}
                print(self.ask_question(self.current_state, [self.machine_name, self.user_name]))
                input_string = self.get_response()
                try:
                    self.all_state_response = self.run_all_regex(input_string)
                    next_state = self.decide_state()
                    response, _class = self.all_state_response.get(next_state)
                    next_state = STATE_TRANSITION_TABLE.get((self.current_state, _class))
                    user_response = response
                except:
                    next_state = 'CONFUSED'
                next_state=self.check_exit(input_string, next_state)
                self.previous_state = self.current_state
                self.current_state = next_state
                continue

            elif self.current_state == 'WANT':
                print(self.ask_question(self.current_state, [self.machine_name, user_response]))
                input_string = self.get_response()
                try:
                    self.all_state_response = self.run_all_regex(input_string)
                    next_state = self.decide_state()
                    response, _class = self.all_state_response.get(next_state)
                    next_state = STATE_TRANSITION_TABLE.get((self.current_state, _class))
                    user_response = response
                except:
                    next_state = 'CONFUSED'
                next_state = self.check_exit( input_string, next_state)
                self.previous_state = self.current_state
                self.current_state = next_state
                continue

            elif self.current_state == 'FEEL':
                print(self.ask_question(self.current_state, [self.machine_name, user_response]))
                input_string = self.get_response()
                try:
                    self.all_state_response = self.run_all_regex(input_string)
                    next_state = self.decide_state()
                    response, _class = self.all_state_response.get(next_state)
                    next_state = STATE_TRANSITION_TABLE.get((self.current_state, _class))
                    user_response = response
                except:
                    next_state = 'CONFUSED'
                next_state=self.check_exit( input_string, next_state)
                self.previous_state = self.current_state
                self.current_state = next_state
                continue

            elif self.current_state == 'HAVE':
                print(self.ask_question(self.current_state, [self.machine_name, user_response]))
                input_string = self.get_response()
                try:
                    self.all_state_response = self.run_all_regex(input_string)
                    next_state = self.decide_state()
                    response, _class = self.all_state_response.get(next_state)
                    next_state = STATE_TRANSITION_TABLE.get((self.current_state, _class))
                    user_response = response
                except:
                    next_state = 'CONFUSED'
                next_state=self.check_exit(input_string, next_state)
                self.previous_state = self.current_state
                self.current_state = next_state
                continue

            elif self.current_state == 'DID':
                print(self.ask_question(self.current_state, [self.machine_name, user_response]))
                input_string = self.get_response()
                try:
                    self.all_state_response = self.run_all_regex(input_string)
                    next_state = self.decide_state()
                    response, _class = self.all_state_response.get(next_state)
                    next_state = STATE_TRANSITION_TABLE.get((self.current_state, _class))
                    user_response = response
                except:
                    next_state = 'CONFUSED'
                next_state=self.check_exit(input_string, next_state)
                self.previous_state = self.current_state
                self.current_state = next_state
                continue


            elif self.current_state == 'CONFUSED':
                print(self.ask_question(self.current_state, [self.machine_name, self.user_name]))
                input_string = self.get_response()
                try:
                    self.all_state_response = self.run_all_regex(input_string)
                    next_state = self.decide_state()
                    response, _class = self.all_state_response.get(next_state)
                    next_state = STATE_TRANSITION_TABLE.get((self.current_state, _class))
                    user_response = response
                except:
                    next_state = 'CONFUSED'
                next_state=self.check_exit(input_string, next_state)
                self.current_state = next_state
                continue

            elif self.current_state == 'EXIT':
                print(self.ask_question(self.current_state, [self.machine_name, self.user_name]))
                break

            elif self.current_state == 'DECIDE_STATE':
                ''' Set only the next state to process does nothing else'''
                self.previous_state = 'DECIDE_STATE'
                self.current_state = self.decide_state()
                item = self.all_state_response.get(self.current_state)
                if item: 
                    user_response = item[0]                
                continue


if __name__ == '__main__':
    current_state = 'GREET'
    agent_name = 'Eliza'
    orange_bot = Machine(agent_name, current_state)
    orange_bot.run()
