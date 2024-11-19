;Author: Bruce Chidley
;PHIL 870 Individual Project


extensions [
             gis ;Used to read in geographical information
             csv ;Used to read in turtle information
]

globals [
          kingston-residential ;Kingston residential geographical data. All turtles return here at the end of every day
          kingston-commercial ;Kingston commercial geographical data. All turtles go here on the weekends
          kingston-education ;Kingston educational geographical data. Student turtles go here during the day on weekdays
          kingston-work ;Kingston work geographical data. Adult turtles will go here during the day to work on weekdays
          bounds ;Exists for the sake of window size

          time ;Tracks time of the day and day of week (every day is 2 ticks for morning/night)

          total-susceptible-count ;Tracks total number of susceptible turtles on each tick
          total-exposed-count ;Tracks total number of exposed turtles on each tick
          total-infectious-count ;Tracks total number of infectious turtles on each tick
          total-recovered-count ;Tracks total number of recovered turtles on each tick

          masked-count ;Tracks the number of masked agents
          vaccinated-count ;Tracks the number of vaccinated agents
]

turtles-own[
             home-patch ;Home patch that they return to on each even tick
             work-patch ;Work patch that they go to during the day
             store-1-patch ;Store-1 patch that they go to on one day of the weekend
             store-2-patch ;Store-2 patch that they go to on another day of the weekend
             age-bracket ;Age bracket for vaccination rates

             susceptible ;True/False if turtle is susceptible
             exposed ;True/False if turtle is exposed
             infectious ;True/False if turtle is infectious
             recovered ;True/False if turtle is recovered

             time-in-class ;Tracks the time an turtle has been in a class
             total-time-in-class ;Tracks the total time that an turtle will remain in a class (calculated when an turtle enters class)

             isolating ;True/False if turtle is isolating (can only happen when infectious)
             mask-factor ;Multiplies the chance of infection by mask-factor, between 0 and 1
             vaccine-factor ;Multiplies the chance of infection by vaccine-factor, between 0 and 1

             bad-actor ;True/False if an turtle is a "bad actor" (i.e. intentionally or unintentionally spreads disease at an increased rate)
]


;Runs when setup button is pressed
to setup

  clear-all ;Clears all data from previous instances

  setup-world ;Reads in all Kingston geographical data

  read-turtles ;Reads in all pre-generated turtles and assigns variable values

  assign-bad-actors ;Makes some turtles bad actors at some user-defined probability

  assign-classes ;Assigns turtles to classes (SEIR) and masks/vaccinates turtles based on real data

  update-numbers ;Updates global variables for tracking purposes

  reset-ticks ;Sets ticks back to 0

end


;Occurs every tick
to go

  clock ;Update the time of day

  ;On even time steps, make all turtles go home and spread disease
  if member? time [0 2 4 6 8 10 12][

    ask turtles [
      move-to home-patch
    ]

    ;For all residential patches (i.e. where all turtles are), spread disease
    ask patches gis:intersecting kingston-residential [

      spread-disease

    ]

  ]

  ;On weekday time steps (second time step of the day for 5 days in a row to mimic Monday, Tuesday, etc.), make turtles go to work and spread disease
  if member? time [1 3 5 7 9][

    ask turtles [
      move-to work-patch
    ]

    ;For all work patches, including schools, spread disease
    ask patches gis:intersecting kingston-work [

      spread-disease

    ]

  ]

  ;On one weekend day on the second time step, make turtles go to the closest store to them and spread disease
  if time = 11[

    ask turtles [
      move-to store-1-patch
    ]

    ;For all commercial patches, spread disease
    ask patches gis:intersecting kingston-commercial [

      spread-disease

    ]

  ]

  ;On the other weekend day on the second time step, make turtles go to the second closest store to them and spread disease
  if time = 13[

    ask turtles [
      move-to store-2-patch
    ]

    ;For all commerical patches, spread disease
    ask patches gis:intersecting kingston-commercial [

      spread-disease

    ]

  ]

  update-turtle-states ;Updates turtle variables (moves them through SEIR classes based on time)

  update-numbers ;Updates global variables

  tick ;Increase tick by 1

end


;Reads in Kingston geographical information
to setup-world

  ;The bounds shapefile (containing the bottom left and top right points only) is in order to fix the scaling. Not sure how to do it otherwise, but coordinates seem to be strangely scaled otherwise
  set bounds gis:load-dataset "bounds.shp"

  ;Reads in all kingston information
  set kingston-residential gis:load-dataset "kingston_residential.shp"
  set kingston-commercial gis:load-dataset "kingston_commercial.shp"
  set kingston-education gis:load-dataset "kingston_education.shp"
  set kingston-work gis:load-dataset "kingston_work.shp"

  ;Resizes the world to be the max/min values in the bounds file
  resize-world (gis:property-minimum bounds "x_cord") (gis:property-maximum bounds "x_cord") (gis:property-minimum bounds "y_cord") (gis:property-maximum bounds "y_cord")

  ;Join them all together
  gis:set-world-envelope (gis:envelope-union-of (gis:envelope-of bounds)
                                                (gis:envelope-of kingston-residential)
                                                (gis:envelope-of kingston-commercial)
                                                (gis:envelope-of kingston-education)
                                                (gis:envelope-of kingston-work))

  ;Colours the patches colours based on the patch (building) type
  gis:set-drawing-color red
  gis:draw kingston-residential 1

  gis:set-drawing-color blue
  gis:draw kingston-commercial 1

  gis:set-drawing-color yellow
  gis:draw kingston-education 1

  gis:set-drawing-color green
  gis:draw kingston-work 1

end


;Read turtle data in
to read-turtles

  ;Opens CSV file containing turtle information
  file-open "turtle_data.csv"

  ;Loops until all rows are read
  while [ not file-at-end? ] [

    let data csv:from-row file-read-line

    ;Creates turtles one at a time
    create-turtles 1 [
      set xcor    item 7 data
      set ycor    item 8 data
      set size    3
      set color   orange
      set heading 100
      set home-patch patch (item 7 data) (item 8 data)
      set work-patch patch (item 9 data) (item 10 data)
      set store-1-patch patch (item 11 data) (item 12 data)
      set store-2-patch patch (item 13 data) (item 14 data)
      set age-bracket item 2 data
    ]
  ]

  ;Closes CSV file
  file-close

end


;Assigns turtles classes at start of simulation
to assign-classes

  ask turtles[

    ;isolating will be false, so that the infection can get off the ground more easily
    set isolating false

    let chance-infectious random-float 1

    ;If the random float between 0 and 1 is less than or equal to the user-defined value, make the turtle infectious
    (ifelse  ((chance-infectious >= 0) and (chance-infectious <= 0.4))[

      set susceptible true
      set exposed false
      set infectious false
      set recovered false

      ;Time in class will be 999 if an turtle is susceptible (as they do not move into other classes naturally)
      set total-time-in-class 999

    ]((chance-infectious > 0.4) and (chance-infectious <= 0.6)) [

      set susceptible false
      set exposed true
      set infectious false
      set recovered false

      ;Sets total time in class per the user-specified value
      set total-time-in-class (random-normal mean-exposed-time exposed-time-std)
   ] ((chance-infectious > 0.6) and (chance-infectious <= 0.8)) [

      set susceptible false
      set exposed false
      set infectious true
      set recovered false

      ;Sets total time in class per the user-specified value
      set total-time-in-class (random-normal mean-infectious-time infectious-time-std)
   ] ((chance-infectious > 0.8) and (chance-infectious <= 1)) [

      set susceptible false
      set exposed false
      set infectious false
      set recovered true

      ;Sets total time in class per the user-specified value
      set total-time-in-class (random-normal mean-recovered-time recovered-time-std)
   ])


    let mask-prob (random-float 1)

    ;Masks turtle if the random float is less than 0.7 and they are not a bad actor
    ifelse (mask-prob <= mask-chance) and (not (bad-actor and (not bad-actor-mask))) [

     ;Effectiveness of the mask is equal to the user-specified value
     set mask-factor mask-infection-factor

    ][

      set mask-factor 1

    ]


    let prob-vaccinated random-float 1

    set vaccine-factor 1

    ;Can only vaccinate if the turtle is not a bad actor
    if (not (bad-actor and (not bad-actor-vaccine)))[

      ;Vaccinates turtles based on their age bracket (values estimated from real data)
      (ifelse age-bracket = 0 [

        if prob-vaccinated <= 0.251 [

          set vaccine-factor vaccine-infection-factor

        ]

      ] age-bracket = 1 [

        if prob-vaccinated <= 0.771 [

          set vaccine-factor vaccine-infection-factor

        ]

      ] age-bracket = 2 [

        if prob-vaccinated <= 0.819 [

          set vaccine-factor vaccine-infection-factor

        ]

      ] age-bracket = 3 [

        if prob-vaccinated <= 0.851 [

          set vaccine-factor vaccine-infection-factor

        ]

      ] age-bracket = 4 [

        if prob-vaccinated <= 0.883 [

          set vaccine-factor vaccine-infection-factor

        ]

      ] age-bracket = 5 [

        if prob-vaccinated <= 0.885 [

          set vaccine-factor vaccine-infection-factor

        ]

      ] age-bracket = 6 [

        if prob-vaccinated <= 0.940 [

          set vaccine-factor vaccine-infection-factor

        ]

      ] age-bracket = 7 [

        if prob-vaccinated <= 0.982 [

          set vaccine-factor vaccine-infection-factor

        ]

      ] age-bracket = 8 [

        if prob-vaccinated <= 0.990 [

          set vaccine-factor vaccine-infection-factor

        ]

      ]

      )

    ]

    ;Set the time in current class equal to 0
    set time-in-class 0

  ]

end


;Assigns bad actors
to assign-bad-actors

  ask turtles [

    let bad-actor-prob random-float 1

    ;If the random float is less than or equal to the user-specified value
    ifelse bad-actor-prob <= bad-actor-proportion [

      set bad-actor true

    ][

      set bad-actor false

    ]

  ]

end


;Spread disease between turtles at a given patch
to spread-disease

  ;Count the number of susceptible turtles at a given patch for the sake of determining disease spread dynamics (forced by basic reproductive ratio)
  let susceptible-count 0

  ask turtles-here [

    if susceptible [

      set susceptible-count (susceptible-count + 1)

    ]

  ]

  ;Loops over all susceptible turtles at the current patch
  ask turtles-here with [susceptible] [

    ;Temporary variable will be false unless successfully infected by another turtle
    let exposed-success false

    ;Loops over all turtles that are infectious and not isolating (in the real world, isolating people would remain at home, but this accomplishes the same thing)
    ask turtles-here with [infectious and (not isolating)] [

      ;Calcualtes the factor to mutliply the basic infection chance by based on masking/vaccinating
      let masked-vaccinated ((mask-factor * ([mask-factor] of myself)) * ([vaccine-factor] of myself))

      ;Calculates the chance of infection
      let odds (masked-vaccinated * (basic-reproductive-ratio / total-time-in-class) / susceptible-count)

      ;If either the susceptible turtle or the infectious turtle is a bad actor
      ifelse  (([bad-actor] of myself) or bad-actor) [

        ;Try infection based on the strength of the bad actor
        repeat bad-actor-strength [

          let against (random-float 1)

          if against < odds [

            set exposed-success true

          ]

        ]

      ;Otherwise, try infection only once
      ][

        let against (random-float 1)

        if against < odds [

          set exposed-success true

        ]

      ]

    ]

    ;If the infection is successful from any of the turtles, make the susceptible turtle exposed
    if exposed-success [

      become-exposed

    ]

  ]

end


;Make an turtle exposed upon successful infection
to become-exposed

  ;Update classes
  set susceptible false
  set exposed true

  ;Reset time in class, and calculate total time in exposed class
  set time-in-class 0
  set total-time-in-class (random-normal mean-exposed-time exposed-time-std)

end


;Update turtle states on every tick
to update-turtle-states

  ask turtles[

    ;If exposed and the current time in class exceeds the specified total time in class, move the turtle to infectious class
    (ifelse (exposed and (time-in-class > total-time-in-class))[

      set exposed false
      set infectious true

      set time-in-class 0
      set total-time-in-class (random-normal mean-infectious-time infectious-time-std)

      let isolation-prob (random-float 1)

      ;turtle isolates based on user-specified value if they are not a bad actor
      if (isolation-prob <= isolation-rate) and (not (bad-actor and (not bad-actor-isolate))) [

        set isolating true

      ]

    ;If it is time to move into recovered class
    ] (infectious and (time-in-class > total-time-in-class))[

      set infectious false
      set recovered true

      set time-in-class 0
      set total-time-in-class (random-normal mean-recovered-time recovered-time-std)

      set isolating false

    ;If it is time to move from the recovered to the susceptible class
    ] (recovered and (time-in-class > total-time-in-class))[

      set recovered false
      set susceptible true

      set time-in-class 0
      set total-time-in-class 999

    ])

    ;Increase the time in class by 1
    set time-in-class (time-in-class + 1)

  ]

end


;Update global variables on each tick
to update-numbers

  ;Create temporary variables to count state numbers
  let susceptible-temp 0
  let exposed-temp 0
  let infectious-temp 0
  let recovered-temp 0

  let mask-temp 0
  let vaccine-temp 0

  ask turtles[

    (ifelse susceptible [

      set susceptible-temp (susceptible-temp + 1)

    ] exposed [

      set exposed-temp (exposed-temp + 1)

    ] infectious [

      set infectious-temp (infectious-temp + 1)

    ] recovered [

      set recovered-temp (recovered-temp + 1)

    ])

    if mask-factor < 1 [

      set mask-temp (mask-temp + 1)

    ]

    if vaccine-factor < 1 [

      set vaccine-temp (vaccine-temp + 1)

    ]

  ]

  ;Update global variables based on new counts
  set total-susceptible-count susceptible-temp
  set total-exposed-count exposed-temp
  set total-infectious-count infectious-temp
  set total-recovered-count recovered-temp
  set masked-count mask-temp
  set vaccinated-count vaccine-temp

end


;Update time variable
to clock

  ;Counts from 0 to 13 repeatedly
  set time ticks mod 14

end
@#$#@#$#@
GRAPHICS-WINDOW
316
10
1171
690
-1
-1
1.0
1
10
1
1
1
0
0
0
1
-544
302
-566
104
0
0
1
ticks
30.0

BUTTON
32
34
95
67
NIL
setup
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

PLOT
309
712
1428
1104
plot 1
NIL
NIL
0.0
10.0
0.0
10.0
true
false
"" ""
PENS
"default" 1.0 0 -8330359 true "" "plot total-susceptible-count"
"pen-1" 1.0 0 -612749 true "" "plot total-exposed-count"
"pen-2" 1.0 0 -2674135 true "" "plot total-infectious-count"
"pen-3" 1.0 0 -13791810 true "" "plot total-recovered-count"

BUTTON
118
32
181
65
NIL
go
T
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

SLIDER
5
553
304
586
bad-actor-proportion
bad-actor-proportion
0
1
0.0
0.01
1
NIL
HORIZONTAL

INPUTBOX
7
136
144
196
basic-reproductive-ratio
3.0
1
0
Number

INPUTBOX
5
208
142
268
mean-exposed-time
9.0
1
0
Number

INPUTBOX
150
207
288
267
exposed-time-std
2.0
1
0
Number

INPUTBOX
5
283
141
343
mean-infectious-time
16.0
1
0
Number

INPUTBOX
150
284
288
344
infectious-time-std
4.0
1
0
Number

INPUTBOX
150
356
288
416
recovered-time-std
2.0
1
0
Number

INPUTBOX
5
356
140
416
mean-recovered-time
14.0
1
0
Number

SLIDER
5
597
304
630
mask-infection-factor
mask-infection-factor
0
1
0.9
0.01
1
NIL
HORIZONTAL

SLIDER
5
685
303
718
isolation-rate
isolation-rate
0
1
0.3
0.01
1
NIL
HORIZONTAL

SLIDER
5
641
304
674
vaccine-infection-factor
vaccine-infection-factor
0
1
0.5
0.01
1
NIL
HORIZONTAL

SLIDER
6
502
305
535
bad-actor-strength
bad-actor-strength
1
5
1.0
1
1
NIL
HORIZONTAL

SWITCH
7
754
149
787
bad-actor-mask
bad-actor-mask
1
1
-1000

SWITCH
6
794
162
827
bad-actor-vaccine
bad-actor-vaccine
1
1
-1000

SWITCH
6
833
156
866
bad-actor-isolate
bad-actor-isolate
1
1
-1000

INPUTBOX
149
136
289
196
mask-chance
0.5
1
0
Number

@#$#@#$#@
## WHAT IS IT?

(a general understanding of what the model is trying to show or explain)

## HOW IT WORKS

(what rules the agents use to create the overall behavior of the model)

## HOW TO USE IT

(how to use the model, including a description of each of the items in the Interface tab)

## THINGS TO NOTICE

(suggested things for the user to notice while running the model)

## THINGS TO TRY

(suggested things for the user to try to do (move sliders, switches, etc.) with the model)

## EXTENDING THE MODEL

(suggested things to add or change in the Code tab to make the model more complicated, detailed, accurate, etc.)

## NETLOGO FEATURES

(interesting or unusual features of NetLogo that the model uses, particularly in the Code tab; or where workarounds were needed for missing features)

## RELATED MODELS

(models in the NetLogo Models Library and elsewhere which are of related interest)

## CREDITS AND REFERENCES

(a reference to the model's URL on the web if it has one, as well as any other necessary credits, citations, and links)
@#$#@#$#@
default
true
0
Polygon -7500403 true true 150 5 40 250 150 205 260 250

airplane
true
0
Polygon -7500403 true true 150 0 135 15 120 60 120 105 15 165 15 195 120 180 135 240 105 270 120 285 150 270 180 285 210 270 165 240 180 180 285 195 285 165 180 105 180 60 165 15

arrow
true
0
Polygon -7500403 true true 150 0 0 150 105 150 105 293 195 293 195 150 300 150

box
false
0
Polygon -7500403 true true 150 285 285 225 285 75 150 135
Polygon -7500403 true true 150 135 15 75 150 15 285 75
Polygon -7500403 true true 15 75 15 225 150 285 150 135
Line -16777216 false 150 285 150 135
Line -16777216 false 150 135 15 75
Line -16777216 false 150 135 285 75

bug
true
0
Circle -7500403 true true 96 182 108
Circle -7500403 true true 110 127 80
Circle -7500403 true true 110 75 80
Line -7500403 true 150 100 80 30
Line -7500403 true 150 100 220 30

butterfly
true
0
Polygon -7500403 true true 150 165 209 199 225 225 225 255 195 270 165 255 150 240
Polygon -7500403 true true 150 165 89 198 75 225 75 255 105 270 135 255 150 240
Polygon -7500403 true true 139 148 100 105 55 90 25 90 10 105 10 135 25 180 40 195 85 194 139 163
Polygon -7500403 true true 162 150 200 105 245 90 275 90 290 105 290 135 275 180 260 195 215 195 162 165
Polygon -16777216 true false 150 255 135 225 120 150 135 120 150 105 165 120 180 150 165 225
Circle -16777216 true false 135 90 30
Line -16777216 false 150 105 195 60
Line -16777216 false 150 105 105 60

car
false
0
Polygon -7500403 true true 300 180 279 164 261 144 240 135 226 132 213 106 203 84 185 63 159 50 135 50 75 60 0 150 0 165 0 225 300 225 300 180
Circle -16777216 true false 180 180 90
Circle -16777216 true false 30 180 90
Polygon -16777216 true false 162 80 132 78 134 135 209 135 194 105 189 96 180 89
Circle -7500403 true true 47 195 58
Circle -7500403 true true 195 195 58

circle
false
0
Circle -7500403 true true 0 0 300

circle 2
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240

cow
false
0
Polygon -7500403 true true 200 193 197 249 179 249 177 196 166 187 140 189 93 191 78 179 72 211 49 209 48 181 37 149 25 120 25 89 45 72 103 84 179 75 198 76 252 64 272 81 293 103 285 121 255 121 242 118 224 167
Polygon -7500403 true true 73 210 86 251 62 249 48 208
Polygon -7500403 true true 25 114 16 195 9 204 23 213 25 200 39 123

cylinder
false
0
Circle -7500403 true true 0 0 300

dot
false
0
Circle -7500403 true true 90 90 120

face happy
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 255 90 239 62 213 47 191 67 179 90 203 109 218 150 225 192 218 210 203 227 181 251 194 236 217 212 240

face neutral
false
0
Circle -7500403 true true 8 7 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Rectangle -16777216 true false 60 195 240 225

face sad
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 168 90 184 62 210 47 232 67 244 90 220 109 205 150 198 192 205 210 220 227 242 251 229 236 206 212 183

fish
false
0
Polygon -1 true false 44 131 21 87 15 86 0 120 15 150 0 180 13 214 20 212 45 166
Polygon -1 true false 135 195 119 235 95 218 76 210 46 204 60 165
Polygon -1 true false 75 45 83 77 71 103 86 114 166 78 135 60
Polygon -7500403 true true 30 136 151 77 226 81 280 119 292 146 292 160 287 170 270 195 195 210 151 212 30 166
Circle -16777216 true false 215 106 30

flag
false
0
Rectangle -7500403 true true 60 15 75 300
Polygon -7500403 true true 90 150 270 90 90 30
Line -7500403 true 75 135 90 135
Line -7500403 true 75 45 90 45

flower
false
0
Polygon -10899396 true false 135 120 165 165 180 210 180 240 150 300 165 300 195 240 195 195 165 135
Circle -7500403 true true 85 132 38
Circle -7500403 true true 130 147 38
Circle -7500403 true true 192 85 38
Circle -7500403 true true 85 40 38
Circle -7500403 true true 177 40 38
Circle -7500403 true true 177 132 38
Circle -7500403 true true 70 85 38
Circle -7500403 true true 130 25 38
Circle -7500403 true true 96 51 108
Circle -16777216 true false 113 68 74
Polygon -10899396 true false 189 233 219 188 249 173 279 188 234 218
Polygon -10899396 true false 180 255 150 210 105 210 75 240 135 240

house
false
0
Rectangle -7500403 true true 45 120 255 285
Rectangle -16777216 true false 120 210 180 285
Polygon -7500403 true true 15 120 150 15 285 120
Line -16777216 false 30 120 270 120

leaf
false
0
Polygon -7500403 true true 150 210 135 195 120 210 60 210 30 195 60 180 60 165 15 135 30 120 15 105 40 104 45 90 60 90 90 105 105 120 120 120 105 60 120 60 135 30 150 15 165 30 180 60 195 60 180 120 195 120 210 105 240 90 255 90 263 104 285 105 270 120 285 135 240 165 240 180 270 195 240 210 180 210 165 195
Polygon -7500403 true true 135 195 135 240 120 255 105 255 105 285 135 285 165 240 165 195

line
true
0
Line -7500403 true 150 0 150 300

line half
true
0
Line -7500403 true 150 0 150 150

pentagon
false
0
Polygon -7500403 true true 150 15 15 120 60 285 240 285 285 120

person
false
0
Circle -7500403 true true 110 5 80
Polygon -7500403 true true 105 90 120 195 90 285 105 300 135 300 150 225 165 300 195 300 210 285 180 195 195 90
Rectangle -7500403 true true 127 79 172 94
Polygon -7500403 true true 195 90 240 150 225 180 165 105
Polygon -7500403 true true 105 90 60 150 75 180 135 105

plant
false
0
Rectangle -7500403 true true 135 90 165 300
Polygon -7500403 true true 135 255 90 210 45 195 75 255 135 285
Polygon -7500403 true true 165 255 210 210 255 195 225 255 165 285
Polygon -7500403 true true 135 180 90 135 45 120 75 180 135 210
Polygon -7500403 true true 165 180 165 210 225 180 255 120 210 135
Polygon -7500403 true true 135 105 90 60 45 45 75 105 135 135
Polygon -7500403 true true 165 105 165 135 225 105 255 45 210 60
Polygon -7500403 true true 135 90 120 45 150 15 180 45 165 90

sheep
false
15
Circle -1 true true 203 65 88
Circle -1 true true 70 65 162
Circle -1 true true 150 105 120
Polygon -7500403 true false 218 120 240 165 255 165 278 120
Circle -7500403 true false 214 72 67
Rectangle -1 true true 164 223 179 298
Polygon -1 true true 45 285 30 285 30 240 15 195 45 210
Circle -1 true true 3 83 150
Rectangle -1 true true 65 221 80 296
Polygon -1 true true 195 285 210 285 210 240 240 210 195 210
Polygon -7500403 true false 276 85 285 105 302 99 294 83
Polygon -7500403 true false 219 85 210 105 193 99 201 83

square
false
0
Rectangle -7500403 true true 30 30 270 270

square 2
false
0
Rectangle -7500403 true true 30 30 270 270
Rectangle -16777216 true false 60 60 240 240

star
false
0
Polygon -7500403 true true 151 1 185 108 298 108 207 175 242 282 151 216 59 282 94 175 3 108 116 108

target
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240
Circle -7500403 true true 60 60 180
Circle -16777216 true false 90 90 120
Circle -7500403 true true 120 120 60

tree
false
0
Circle -7500403 true true 118 3 94
Rectangle -6459832 true false 120 195 180 300
Circle -7500403 true true 65 21 108
Circle -7500403 true true 116 41 127
Circle -7500403 true true 45 90 120
Circle -7500403 true true 104 74 152

triangle
false
0
Polygon -7500403 true true 150 30 15 255 285 255

triangle 2
false
0
Polygon -7500403 true true 150 30 15 255 285 255
Polygon -16777216 true false 151 99 225 223 75 224

truck
false
0
Rectangle -7500403 true true 4 45 195 187
Polygon -7500403 true true 296 193 296 150 259 134 244 104 208 104 207 194
Rectangle -1 true false 195 60 195 105
Polygon -16777216 true false 238 112 252 141 219 141 218 112
Circle -16777216 true false 234 174 42
Rectangle -7500403 true true 181 185 214 194
Circle -16777216 true false 144 174 42
Circle -16777216 true false 24 174 42
Circle -7500403 false true 24 174 42
Circle -7500403 false true 144 174 42
Circle -7500403 false true 234 174 42

turtle
true
0
Polygon -10899396 true false 215 204 240 233 246 254 228 266 215 252 193 210
Polygon -10899396 true false 195 90 225 75 245 75 260 89 269 108 261 124 240 105 225 105 210 105
Polygon -10899396 true false 105 90 75 75 55 75 40 89 31 108 39 124 60 105 75 105 90 105
Polygon -10899396 true false 132 85 134 64 107 51 108 17 150 2 192 18 192 52 169 65 172 87
Polygon -10899396 true false 85 204 60 233 54 254 72 266 85 252 107 210
Polygon -7500403 true true 119 75 179 75 209 101 224 135 220 225 175 261 128 261 81 224 74 135 88 99

wheel
false
0
Circle -7500403 true true 3 3 294
Circle -16777216 true false 30 30 240
Line -7500403 true 150 285 150 15
Line -7500403 true 15 150 285 150
Circle -7500403 true true 120 120 60
Line -7500403 true 216 40 79 269
Line -7500403 true 40 84 269 221
Line -7500403 true 40 216 269 79
Line -7500403 true 84 40 221 269

wolf
false
0
Polygon -16777216 true false 253 133 245 131 245 133
Polygon -7500403 true true 2 194 13 197 30 191 38 193 38 205 20 226 20 257 27 265 38 266 40 260 31 253 31 230 60 206 68 198 75 209 66 228 65 243 82 261 84 268 100 267 103 261 77 239 79 231 100 207 98 196 119 201 143 202 160 195 166 210 172 213 173 238 167 251 160 248 154 265 169 264 178 247 186 240 198 260 200 271 217 271 219 262 207 258 195 230 192 198 210 184 227 164 242 144 259 145 284 151 277 141 293 140 299 134 297 127 273 119 270 105
Polygon -7500403 true true -1 195 14 180 36 166 40 153 53 140 82 131 134 133 159 126 188 115 227 108 236 102 238 98 268 86 269 92 281 87 269 103 269 113

x
false
0
Polygon -7500403 true true 270 75 225 30 30 225 75 270
Polygon -7500403 true true 30 75 75 30 270 225 225 270
@#$#@#$#@
NetLogo 6.4.0
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
<experiments>
  <experiment name="experiment" repetitions="1" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="200"/>
    <metric>total-susceptible-count</metric>
    <metric>total-exposed-count</metric>
    <metric>total-infectious-count</metric>
    <metric>total-recovered-count</metric>
  </experiment>
</experiments>
@#$#@#$#@
@#$#@#$#@
default
0.0
-0.2 0 0.0 1.0
0.0 1 1.0 0.0
0.2 0 0.0 1.0
link direction
true
0
Line -7500403 true 150 150 90 180
Line -7500403 true 150 150 210 180
@#$#@#$#@
0
@#$#@#$#@
