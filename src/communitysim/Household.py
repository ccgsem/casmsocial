from .Place import Place
from .Calendar import Calendar
from .Parameters import Parameters

from typing import Dict
from repast4py.space import DiscretePoint as dpt

from .InsuranceProvider import shopInsuranceProviders

class Household(Place):
    def __init__(self, initDict: Dict):
        placeId = initDict['sp_id']
        location = dpt(x=int(initDict['x']), y=int(initDict['y']), z=0)
        super().__init__(placeId, location)

        self.hasInsurance = initDict['has_hazard_insurance'] == '1'
        self.isOwner = initDict['occupancy_status'] == 'owner_occupied'
        self.hasMortgage = initDict['owner_costs_with_mortgage'] != 'not_applicable'
        
        self.insurancePurchaseData = int(initDict['ins_purchase_date']) if self.hasInsurance else -1


    
    def step(self, calendar: Calendar, rng):
        if len(self.peopleAtPlace) != 0:
            self.perceivedRisk = self.peopleAtPlace[0].risk
        if calendar.isNewMonth:
            self.shopForInsurance(rng)
            self.reduceFuel()

    def shopForInsurance(self, rng):
        if self.hasInsurance:
            return
        if not self.isOwner:
            return
        if self.hasMortgage:
            shopInsuranceProviders(self)
            return

        pShop = rng.random()
        if self.perceivedRisk < Parameters.percievedRiskL:
            if pShop < Parameters.shopPL:
                shopInsuranceProviders(self)
        elif self.perceivedRisk < Parameters.perceivedRiskM:
            if pShop < Parameters.shopPM:
                shopInsuranceProviders(self)
        elif self.perceivedRisk < Parameters.perceivedRiskH:
            if pShop < Parameters.shopPH:
                shopInsuranceProviders(self)
        

    def reduceFuel(self):
        pass

    def purchaseInsurance(self, offers):
        pass
