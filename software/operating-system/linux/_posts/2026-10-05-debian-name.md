---
layout: post
title: "Where Debian Release Codenames Come From"
date: 2026-10-05
categories: [linux]
tags: [linux, debian]
---

## Overview

Package pages on `packages.debian.org` list suites such as `bullseye`, `bookworm`,
`trixie`, `forky` and `sid`. These names were not chosen for their English
meaning. `bullseye` is not a target, and `bookworm` is not a person who reads a
lot. Every Debian codename so far is a character from Pixar's *Toy Story* films.

## Why Toy Story

Debian 1.1 `buzz`, released on June 17, 1996, was the first release with a
codename. By then, Bruce Perens had replaced Ian Murdock as project leader. He
was also working at Pixar, the studio that made the films. The Debian FAQ
credits him with choosing Toy Story names.

## Why codenames exist

Releases have codenames for practical reasons:

- While a release is in development, it has a codename but no version number yet.
- Codenames make mirroring easier. If a real directory such as `unstable` were
  renamed to `stable`, mirrors would have to download a large amount of data again.
- The release-state names are symbolic links:
  - `stable` points to `trixie` (Debian 13).
  - `testing` points to `forky`.
  - `unstable` always points to `sid`.

## All codenames

| Version | Codename | Released | Toy Story character |
|---|---|---|---|
| 1.1 | buzz | 1996 | Buzz Lightyear, the spaceman |
| 1.2 | rex | 1996 | The tyrannosaurus |
| 1.3 | bo | 1997 | Bo Peep, the shepherdess |
| 2.0 | hamm | 1998 | The piggy bank |
| 2.1 | slink | 1999 | Slinky Dog |
| 2.2 | potato | 2000 | Mr. Potato Head |
| 3.0 | woody | 2002 | The cowboy |
| 3.1 | sarge | 2005 | Sergeant of the Green Plastic Army Men |
| 4.0 | etch | 2007 | The Etch-a-Sketch whiteboard |
| 5.0 | lenny | 2009 | The toy binoculars |
| 6 | squeeze | 2011 | The three-eyed aliens |
| 7 | wheezy | 2013 | Rubber penguin with a red bow tie |
| 8 | jessie | 2015 | The yodeling cowgirl |
| 9 | stretch | 2017 | Rubber octopus with suckers on her arms |
| 10 | buster | 2019 | Andy's pet dog |
| 11 | bullseye | 2021 | Woody's toy horse |
| 12 | bookworm | 2023 | Green toy worm with a built-in flashlight |
| 13 | trixie | 2025 | Blue plastic triceratops |
| 14 | forky | TBA | Spork toy made by Bonnie (*Toy Story 4*) |
| 15 | duke | TBA | Duke Caboom, Canadian daredevil toy (*Toy Story 4*) |
| — | sid | never | The neighbor kid who breaks toys |

As of this writing, the Debian FAQ does not describe `forky` and `duke`. In the
table, those two rows are matched to the characters with the same names in
*Toy Story 4*.

## Why unstable is sid

In the first film, Sid Phillips is the boy next door who destroys toys. In
Debian, `sid` is the unstable distribution, where most packages are uploaded
first. It is never released directly; packages must pass through `testing`
before they can reach `stable`. That is why `sid` is the only codename that
never moves to a version number.

## References

- [Debian FAQ, Chapter 6: The Debian archives](https://www.debian.org/doc/manuals/debian-faq/ftparchives.en.html)
  explains the character behind each codename through trixie, why codenames
  exist, the stable/testing/unstable symlinks, and the role of `sid`. It also
  credits Bruce Perens with the Toy Story naming decision.
- [Debian Releases](https://www.debian.org/releases/) provides the release
  dates, the current stable (trixie) and testing (forky) releases, and the
  announced Debian 15 codename `duke`.
- [A Brief History of Debian, Chapter 3: Debian Releases](https://www.debian.org/doc/manuals/project-history/releases.en.html)
  states that 1.1 `buzz` was the first codenamed release and that Bruce Perens
  was working at Pixar at that time.
- [List of Toy Story characters (Wikipedia)](https://en.wikipedia.org/wiki/List_of_Toy_Story_characters)
  describes Forky, Duke Caboom, and Sid Phillips.
