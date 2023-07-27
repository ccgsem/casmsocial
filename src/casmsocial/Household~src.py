from .Place import Place
from .Calendar import Calendar
from .Parameters import Parameters

from typing import Dict
from repast4py.space import DiscretePoint as dpt

from .InsuranceProvider import shopInsuranceProviders
from .Parameters import Parameters

class Household(Place):
    def __init__(self, initDict: Dict):
        placeId = initDict['sp_id']
        location = dpt(x=int(initDict['x']), y=int(initDict['y']), z=0)
        super().__init__(placeId, location)

        self.hasInsurance = initDict['has_hazard_insurance'] == '1'
        self.isOwner = initDict['occupancy_status'] == 'owner_occupied'
        self.hasMortgage = initDict['owner_costs_with_mortgage'] != 'not_applicable'
        
        self.insurancePurchaseData = int(initDict['ins_purchase_date']) if self.hasInsurance else -1

        #### Bianica Additions
        self.fuelReductionLevel = 'none' ## Need to replace with actual current level of fuel reduction from properties file
        self.belowPoverty = initDict['below_poverty'] == 'TRUE'
        self.reasonNoInsurance = 'denied' ## Need to replace with reason household may not have insurance (denied,unaffordable,did_not_shop)
        self.structureLossProbability = 0.89 ## Need to replace with properties structure loss probability from properties file
        ####

        self.structVal = 0 # the value of the structure, need to replace with value from properties file
        self.structYearBuilt = 2000 # the year the structure was built, need to replace with value from properties file


    
    def step(self, calendar: Calendar, rng):
        if len(self.peopleAtPlace) != 0:
            self.perceivedRisk = self.peopleAtPlace[0].risk
        if calendar.isNewMonth:
            self.shopForInsurance(rng)
            self.reduceFuel()

    def shopForInsurance(self, rng):

        query = (self.hasInsurance, self.isOwner, self.hasMortgage, self.perceivedRisk)

        action = decisionRuleMap['shop_for_insurance'].get(query)

        if action == None:
            # This should happen, not every case is in the csv file and the ones that aren't there are when the
            # household already has insurance
            return
        
        pShop = rng.random()

        if action == "100%":
            shopInsuranceProviders(self)
        elif action == "shopPH" and self.perceivedRisk < Parameters.perceivedRiskH and pShop < Parameters.shopPH:
            shopInsuranceProviders(self)
        elif action == "shopPM" and self.perceivedRisk < Parameters.perceivedRiskM and pShop < Parameters.shopPM:
            shopInsuranceProviders(self)
        elif action == "shopPL" and self.perceivedRisk < Parameters.perceivedRiskL and pShop < Parameters.shopPL:
            shopInsuranceProviders(self)
        elif action == "0%":
            return
        else:
            return

    def shopInsuranceProvider(self):

        acceptedProviders = {}
        
        for provider in insuranceProviderMap:
            # Check the premiums from each insurer
            # To reduce computation, I could just select 5 insurance providers at random here to choose from.
            # Then I am not going through 70 providers, which a normal person wouldn't do
            premium = provider.provide_insurance()
            if premium != None:
                acceptedProviders[provider.id] = (provider, premium)

        return acceptedProviders
    
    def reduceFuel(self):

        canAffordFuelReduction = "yes"

        if self.reasonNoInsurance == "unaffordable":
            canAffordFuelReduction = "no"  

        query = (self.hasInsurance, self.reasonNoInsurance, self.isOwner, self.hasMortgage, self.perceivedRisk, self.fuelReductionLevel, canAffordFuelReduction)

        action = decisionRuleMap['shop_for_insurance'].get(query)

        if action == None:
            # This should happen, not every case is in the csv file and the ones that aren't there are when the
            # household already has insurance
            return

        #### Bianica Additions
        if action == "0%":
            return

        pReduce = rng.random()

        if action == "fuelPL" and pReduce < Parameters.fuelPL:
            reduceFuelAtProperty(self)
        elif action == "fuelPM" and pReduce < Parameters.fuelPM:
            reduceFuelAtProperty(self)
        elif action == "fuelPH" and pReduce < Parameters.fuelPH:
            reduceFuelAtProperty(self)
        else:
            return # Not reducing that fuel
        
    #### Bianica Additions
    def reduceFuelAtProperty(self):
        # update fuel reduction level of home
        if self.fuelReductionLevel == 'none':
            self.fuelReductionLevel = 'light'
        elif self.fuelReductionLevel == 'light':
            self.fuelReductionLevel = 'heavy'
        elif self.fuelReductionLevel == 'heavy':
            self.fuelReductionLevel = 'full'
        
        # update probability of structure loss given the increased level of fuel reduction
        if self.fuelReductionLevel == 'light':
            if self.structureLossProbability > 0.89:
                    self.structureLossProbability = 0.89
        elif self.fuelReductionLevel == 'heavy':
            self.structureLossProbability = 0.55
        elif self.fuelReductionLevel == 'full':
            self.structureLossProbability = 0.89
    ####

    def purchaseInsurance(self, offers: dict, rng):

        at_least_one_provider = "no"
        if len(offers) > 0:
            at_least_one_provider = "yes"

        # Need an affordability test to see whether they can afford insurance
        # Setting default to yes right now
        can_afford = "yes"

        query = (at_least_one_provider, can_afford)

        action = decisionRuleMap['purchase_insurance'].get(query)
        
        if action == None:
            return None # This would be an error, need error handling
        if action == "0%":
            self.hasInsurance = False
            return None

        pPurchase = rng.random()
        if (action == "purchasePH" and pPurchase < Parameters.purchasePH) or (action == "purchasePL" and pPurchase < Parameters.purchasePL):

            minPremium = 1000000
            insurerID = 1000000
            provider = None

            # TO DO: IMPLEMENT ME
            # Need to add more criteria that a person would choose insurance on
            # i.e. maybe if their perceived risk is high, they would be more likely to choose a more expensive plan
            # Or if they are very wealthy, they would choose an expensive plan

            for key, value in offers.items():
                if value[1] < minPremium:
                    minPremium = value[1]
                    provider = value[0]
                    insurerID = key

            # On this return, figure out a way to put the fact that they own insurance from a certain insurerID
            # into the ledger somewhere.
            provider["householdsInsured"].append(self) # somehow append an ID to the list of households that an insurance provider has in its portfolio
            self.hasInsurance = True

            # DO something with the self.insurancePurchaseData variable here

            return insurerID, minPremium

        else:
            self.hasInsurance = False
            return None


        
